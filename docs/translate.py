import argparse
import asyncio
import logging
import multiprocessing
import os
import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    from plamo_translate.clients.translate import MCPClient
    from plamo_translate.main import check_server_running
    from plamo_translate.main import start_mcp_server
    from plamo_translate.main import wait_for_server_ready
    from plamo_translate.servers.utils import update_config
except ImportError:
    logger.warning("plamo-translate is not installed")
    MCPClient = None


async def get_translation(
    client: MCPClient, messages: List[Dict[str, str]]  # type: ignore
) -> str:
    async for result in client.translate(messages):
        continue
    return result


def create_vocabulary_format(word_pairs: List[Tuple[str, str]]) -> str:
    vocab_lines = []
    for en_word, ja_word in word_pairs:
        vocab_lines.append(f"<dt>{en_word}<dd>{ja_word}")
    return "\n".join(vocab_lines)


def translate_text(
    client: MCPClient,  # type: ignore
    from_lang: str,
    to_lang: str,
    input_text: str,
    context_text: Optional[str] = None,
    vocabularies: Optional[List[Tuple[str, str]]] = None,
) -> str:
    if vocabularies is None:
        vocabularies = []
    messages = []
    if context_text:
        messages.append(
            {
                "role": "user",
                "content": f"context\n{context_text}",
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"input lang={from_lang}\n{input_text}",
        }
    )
    if len(vocabularies) > 0:
        vocab_content = (
            f"vocabularies lang={from_lang}|{to_lang} order=shuffled\n"
            f"{create_vocabulary_format(vocabularies)}"
        )
        messages.append(
            {
                "role": "user",
                "content": vocab_content,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"output lang={to_lang}\n",
        }
    )
    logger.debug(f"Translation messages: {messages}")
    outputs = asyncio.run(get_translation(client, messages))
    if outputs.endswith("\n"):
        outputs = outputs[:-1]
    return outputs


def parse_po_file(file_path: Path) -> List[Tuple[str, str, str]]:
    """
    .poファイルを解析してmsgid, msgstr, contextを抽出
    Returns: [(msgid, msgstr, context), ...]
    """
    entries = []
    current_msgid = ""
    current_msgstr = ""
    current_context = ""
    in_msgid = False
    in_msgstr = False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # コメント行やコンテキスト情報
                if line.startswith("#"):
                    if line.startswith("#:"):
                        current_context = line[2:].strip()
                    continue

                # msgid行の開始
                if line.startswith("msgid "):
                    if current_msgid and current_msgstr is not None:
                        entry = (current_msgid, current_msgstr, current_context)
                        entries.append(entry)
                    current_msgid = line[6:].strip(' "')
                    current_msgstr = ""
                    current_context = ""
                    in_msgid = True
                    in_msgstr = False
                    continue

                # msgstr行の開始
                if line.startswith("msgstr "):
                    current_msgstr = line[7:].strip(' "')
                    in_msgid = False
                    in_msgstr = True
                    continue

                # 継続行の処理
                if line.startswith('"') and line.endswith('"'):
                    content = line[1:-1]
                    if in_msgid:
                        current_msgid += content
                    elif in_msgstr:
                        current_msgstr += content

            # 最後のエントリを追加
            if current_msgid and current_msgstr is not None:
                entries.append((current_msgid, current_msgstr, current_context))

    except Exception as e:
        logger.error(f"ファイル解析エラー {file_path}: {e}")

    return entries


def is_code_block(msgid: str) -> bool:
    """msgidがコードブロック（doctest等）かどうかを判定"""
    return msgid.lstrip().startswith(">>>")


def should_translate(msgid: str, msgstr: str, force: bool = False) -> bool:
    """翻訳が必要かどうかを判定"""
    # 空のmsgidはスキップ
    if not msgid.strip():
        return False

    # 強制モードでない場合、既に翻訳済みの場合はスキップ
    if not force and msgstr.strip():
        return False

    # フィールド本体がフラット化された複合メッセージ（例:
    # "A named tuple of ...:      - ``values`` ..."）は、構造化された
    # 個別エントリ側で翻訳されるため、翻訳するとリスト構造が失われる。
    # 未翻訳のままにしておく必要がある。
    if re.search(r":\s{2,}-\s", msgid):
        return False

    # 特定のパターンはスキップ
    skip_patterns = [
        r"^:.*:$",  # :ref:`genindex` など
        r"^\.\..*$",  # .. directive など
        r"^\s*$",  # 空白のみ
        r"^[a-z0-9_\\.]+$",  # 関数名などの識別子（例: apply\_for\_axis）
    ]

    for pattern in skip_patterns:
        if re.match(pattern, msgid):
            return False

    return True


def translate_code_block_comments(
    client: MCPClient,  # type: ignore
    from_lang: str,
    to_lang: str,
    msgid: str,
    context_text: Optional[str] = None,
) -> str:
    """コードブロック内の ``#`` コメントのみを翻訳

    コード自体を翻訳するとdoctestが壊れるため、各行の ``#`` 以降の
    コメント部分だけを翻訳して置き換える。msgidは.poファイルの
    エスケープ表現（改行は ``\\n``）のまま処理する。
    """
    lines = msgid.split("\\n")
    translated_lines = []
    for line in lines:
        comment_index = line.find("#")
        comment = line[comment_index + 1 :].strip() if comment_index >= 0 else ""
        if comment_index < 0 or not re.search(r"[A-Za-z]", comment):
            translated_lines.append(line)
            continue
        translated = translate_text(
            client, from_lang, to_lang, comment, context_text=context_text
        )
        # .poファイルの二重引用符エスケープを維持する
        translated = translated.replace('"', '\\"')
        translated_lines.append(line[: comment_index + 1] + " " + translated)
    return "\\n".join(translated_lines)


def is_file_fully_translated(file_path: Path) -> bool:
    """ファイルが完全に翻訳済みかどうかを判定"""
    entries = parse_po_file(file_path)

    # 翻訳対象エントリを取得
    translatable_entries = [
        (msgid, msgstr, context)
        for msgid, msgstr, context in entries
        if should_translate(msgid, msgstr, force=False)
    ]

    return len(translatable_entries) == 0


def write_po_file(file_path: Path, entries: List[Tuple[str, str, str]]) -> None:
    """翻訳済みの内容を.poファイルに書き込み"""
    try:
        # 元のファイルを読み込んで形式を保持
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        # 翻訳辞書を作成
        translation_dict = {
            msgid: msgstr for msgid, msgstr, _ in entries if msgstr
        }

        # 新しい内容を生成
        new_lines = []
        current_msgid = ""
        in_msgid = False

        for line in original_lines:
            if line.strip().startswith("msgid "):
                current_msgid = line.strip()[6:].strip(' "')
                in_msgid = True
                new_lines.append(line)
                continue
            elif line.strip().startswith("msgstr "):
                if current_msgid in translation_dict:
                    # 翻訳済みの場合は置換
                    translated = translation_dict[current_msgid]
                    new_lines.append(f'msgstr "{translated}"\n')
                else:
                    new_lines.append(line)
                in_msgid = False
                continue
            elif line.strip().startswith('"') and line.strip().endswith('"'):
                # 継続行の処理
                if in_msgid:
                    current_msgid += line.strip()[1:-1]
                new_lines.append(line)
            else:
                new_lines.append(line)

        # ファイルに書き込み
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    except Exception as e:
        logger.error(f"ファイル書き込みエラー {file_path}: {e}")


def translate_po_file_with_lang(
    client: MCPClient,  # type: ignore
    file_path: Path,
    from_lang: str,
    to_lang: str,
    global_context: Optional[str] = None,
    force: bool = False,
) -> None:
    """単一の.poファイルを翻訳"""
    logger.info(f"翻訳中: {file_path}")

    # 強制モードでない場合、ファイルが完全に翻訳済みかチェック
    if not force and is_file_fully_translated(file_path):
        logger.info(f"  スキップ: 既に翻訳済み {file_path}")
        return

    # .poファイルを解析
    entries = parse_po_file(file_path)

    # 翻訳対象を抽出
    to_translate = [
        (msgid, msgstr, context)
        for msgid, msgstr, context in entries
        if should_translate(msgid, msgstr, force)
    ]

    if not to_translate:
        logger.info(f"  翻訳対象なし: {file_path}")
        return

    force_msg = "(強制モード)" if force else ""
    logger.info(f"  翻訳対象: {len(to_translate)}件 {force_msg}")

    # 同じファイル内の翻訳対象テキストをcontextとして収集
    # （コードブロックは長大になるため除外）
    file_context_texts = [
        msgid
        for msgid, _, _ in to_translate
        if msgid.strip() and not is_code_block(msgid)
    ]

    # グローバルcontextとファイル内contextを結合
    context_parts = []
    if global_context:
        context_parts.append(global_context)

    if file_context_texts:
        file_context = "翻訳対象テキスト:\n" + "\n".join(
            f"- {text}" for text in file_context_texts[:10]
        )  # 最初の10件まで
        context_parts.append(file_context)

    combined_context = "\n\n".join(context_parts) if context_parts else None

    # 翻訳実行
    translated_entries = []
    for i, (msgid, msgstr, context) in enumerate(to_translate, 1):
        logger.debug(f"  {i}/{len(to_translate)}: {msgid[:50]}...")
        if is_code_block(msgid):
            # コードブロックはコメント部分のみ翻訳
            translated = translate_code_block_comments(
                client,
                from_lang,
                to_lang,
                msgid,
                context_text=combined_context,
            )
        else:
            translated = translate_text(
                client,
                from_lang,
                to_lang,
                msgid,
                context_text=combined_context,
            )
        translated_entries.append((msgid, translated, context))

    # 翻訳結果をマージ
    all_entries = []
    translation_dict = {
        msgid: translated for msgid, translated, _ in translated_entries
    }

    for msgid, msgstr, context in entries:
        if msgid in translation_dict:
            all_entries.append((msgid, translation_dict[msgid], context))
        else:
            all_entries.append((msgid, msgstr, context))

    # ファイルに書き込み
    write_po_file(file_path, all_entries)
    logger.info(f"  完了: {file_path}")


def find_po_files(to_lang: str) -> List[Path]:
    """指定された言語の.poファイルを検索"""
    po_files: List[Path] = []
    base_path = f"docs/locale/{to_lang}/LC_MESSAGES"
    base = Path(base_path)

    if base.exists():
        po_files.extend(base.glob("**/*.po"))

    return sorted(po_files)


def translate_all_po_files(
    client: MCPClient,  # type: ignore
    from_lang: str,
    to_lang: str,
    global_context: Optional[str] = None,
    force: bool = False,
) -> None:
    """全ての.poファイルを翻訳"""
    po_files = find_po_files(to_lang)

    if not po_files:
        logger.warning(
            f".poファイルが見つかりません: docs/locale/{to_lang}/LC_MESSAGES"
        )
        return

    force_msg = "(強制モード)" if force else ""
    logger.info(f"翻訳設定: {from_lang} → {to_lang} {force_msg}")
    logger.info(f"見つかった.poファイル: {len(po_files)}件")

    for i, po_file in enumerate(po_files, 1):
        logger.info(f"\n[{i}/{len(po_files)}] {po_file}")
        translate_po_file_with_lang(
            client, po_file, from_lang, to_lang, global_context, force
        )

    logger.info("\n全ての翻訳が完了しました！")


def check_translation_needed(
    to_lang: str, force: bool = False, dry_run: bool = False
) -> bool:
    """翻訳が必要かどうかを事前チェック"""
    po_files = find_po_files(to_lang)

    if not po_files:
        logger.warning(
            f".poファイルが見つかりません: "
            f"docs/locale/{to_lang}/LC_MESSAGES"
        )
        return False

    translation_needed = False
    total_translatable = 0
    file_details = []

    for po_file in po_files:
        if not force and is_file_fully_translated(po_file):
            if dry_run:
                file_details.append((po_file, 0, "完全に翻訳済み"))
            continue

        entries = parse_po_file(po_file)
        translatable_entries = [
            (msgid, msgstr, context)
            for msgid, msgstr, context in entries
            if should_translate(msgid, msgstr, force)
        ]

        if translatable_entries:
            translation_needed = True
            total_translatable += len(translatable_entries)
            if dry_run:
                file_details.append(
                    (po_file, len(translatable_entries), "翻訳対象あり")
                )
        elif dry_run:
            file_details.append((po_file, 0, "翻訳対象なし"))

    if dry_run:
        logger.info("=== Dry Run モード: 翻訳対象の詳細 ===")
        logger.info(f"見つかった.poファイル: {len(po_files)}件")
        logger.info("")

        for po_file, count, status in file_details:
            # 相対パスに変換（エラー回避のため try-except を使用）
            try:
                relative_path = po_file.relative_to(Path.cwd())
            except ValueError:
                relative_path = po_file

            if count > 0:
                logger.info(f"📝 {relative_path}: {count}件 ({status})")
            else:
                logger.info(f"✅ {relative_path}: {status}")

        logger.info("")
        if translation_needed:
            force_msg = " (強制モード)" if force else ""
            logger.info(f"🔄 合計翻訳対象: {total_translatable}件{force_msg}")
        else:
            logger.info("✅ 翻訳対象のエントリが見つかりませんでした")

        return translation_needed

    if translation_needed:
        force_msg = " (強制モード)" if force else ""
        logger.info(
            f"翻訳対象: {total_translatable}件のエントリが"
            f"見つかりました{force_msg}"
        )
    else:
        logger.info("翻訳対象のエントリが見つかりませんでした")

    return translation_needed


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="PO files translator using plamo-translate"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force translation even if already translated",
    )
    parser.add_argument(
        "--from-lang", default="en", help="Source language (default: en)"
    )
    parser.add_argument(
        "--to-lang", default="ja", help="Target language (default: ja)"
    )
    parser.add_argument(
        "--model-name",
        default="mlx-community/plamo-2-translate",
        help="Model name (default: mlx-community/plamo-2-translate)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check translation targets without actually translating",
    )

    args = parser.parse_args()

    model_name = args.model_name
    backend_type = "mlx"
    from_lang = args.from_lang
    to_lang = args.to_lang
    force = args.force
    dry_run = args.dry_run

    global_context_text = """qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのPythonフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-functionsは、qfevalの中でも、金融時系列データを効率的に扱うための関数群を提供します。"""

    if dry_run:
        logger.info("=== Dry Run モード ===")
    else:
        logger.info("設定:")
        logger.info(f"  モデル: {model_name}")

    logger.info(f"  翻訳: {from_lang} → {to_lang}")
    logger.info(f"  強制モード: {force}")

    # 翻訳対象があるかを事前チェック
    logger.info("翻訳対象をチェック中...")
    has_targets = check_translation_needed(to_lang, force, dry_run)

    if dry_run:
        logger.info("=== Dry Run 完了 ===")
        if has_targets:
            logger.info("終了コード: 1 (翻訳対象あり)")
            exit(1)
        else:
            logger.info("終了コード: 0 (翻訳対象なし)")
            exit(0)

    if not has_targets:
        logger.info("翻訳対象がないため、処理を終了します")
        return

    # 翻訳対象があるのでMLXサーバーを起動
    logger.info("翻訳対象が見つかりました。MLXサーバーを起動します...")

    update_config(backend_type=backend_type, model_name=model_name)
    if "PLAMO_TRANSLATE_CLI_MODEL_NAME" not in os.environ:
        os.environ["PLAMO_TRANSLATE_CLI_MODEL_NAME"] = model_name

    if not check_server_running():
        server = multiprocessing.Process(
            target=start_mcp_server,
            args=(backend_type, "CRITICAL", True),
            daemon=True,
        )
        server.start()
        wait_for_server_ready()

    client = MCPClient(stream=False)
    # .poファイル翻訳を実行
    logger.info("=== .poファイル翻訳開始 ===")
    translate_all_po_files(
        client, from_lang, to_lang, global_context_text, force
    )


if __name__ == "__main__":
    main()
