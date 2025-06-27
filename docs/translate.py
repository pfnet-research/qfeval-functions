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

from plamo_translate.clients import translate
from plamo_translate.main import check_server_running
from plamo_translate.main import start_mcp_server
from plamo_translate.main import wait_for_server_ready
from plamo_translate.servers.utils import update_config

logger = logging.getLogger(__name__)


async def get_translation(
    client: translate.MCPClient, messages: List[Dict[str, str]]
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
    client: translate.MCPClient,
    from_lang: str,
    to_lang: str,
    input_text: str,
    context_text: Optional[str] = None,
    vocabularies: List[Tuple[str, str]] = [],
) -> str:
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
        print(f"ファイル解析エラー {file_path}: {e}")

    return entries


def should_translate(msgid: str, msgstr: str) -> bool:
    """翻訳が必要かどうかを判定"""
    # 空のmsgidはスキップ
    if not msgid.strip():
        return False

    # 既に翻訳済みの場合はスキップ
    if msgstr.strip():
        return False

    # 特定のパターンはスキップ
    skip_patterns = [
        r"^:.*:$",  # :ref:`genindex` など
        r"^\.\..*$",  # .. directive など
        r"^\s*$",  # 空白のみ
    ]

    for pattern in skip_patterns:
        if re.match(pattern, msgid):
            return False

    return True


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
        print(f"ファイル書き込みエラー {file_path}: {e}")


def translate_po_file(client: translate.MCPClient, file_path: Path) -> None:
    """単一の.poファイルを翻訳"""
    print(f"翻訳中: {file_path}")

    # .poファイルを解析
    entries = parse_po_file(file_path)

    # 翻訳対象を抽出
    to_translate = [
        (msgid, msgstr, context)
        for msgid, msgstr, context in entries
        if should_translate(msgid, msgstr)
    ]

    if not to_translate:
        print(f"  翻訳対象なし: {file_path}")
        return

    print(f"  翻訳対象: {len(to_translate)}件")

    # 翻訳実行
    translated_entries = []
    for i, (msgid, msgstr, context) in enumerate(to_translate, 1):
        print(f"  {i}/{len(to_translate)}: {msgid[:50]}...")
        translated = translate_text(client, "en", "ja", msgid)
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
    print(f"  完了: {file_path}")


def find_po_files(base_path: str = "docs/locale/ja/LC_MESSAGES") -> List[Path]:
    """指定されたパス以下の.poファイルを検索"""
    po_files: List[Path] = []
    base = Path(base_path)

    if base.exists():
        po_files.extend(base.glob("**/*.po"))

    return sorted(po_files)


def translate_all_po_files(client: translate.MCPClient) -> None:
    """全ての.poファイルを翻訳"""
    po_files = find_po_files()

    if not po_files:
        print(".poファイルが見つかりません")
        return

    print(f"見つかった.poファイル: {len(po_files)}件")

    for i, po_file in enumerate(po_files, 1):
        print(f"\n[{i}/{len(po_files)}] {po_file}")
        translate_po_file(client, po_file)

    print("\n全ての翻訳が完了しました！")


if __name__ == "__main__":
    model_name = "mlx-community/plamo-2-translate"
    backend_type = "mlx"
    from_lang = "en"
    to_lang = "ja"
    strema = False
    context_text = """qfevalは、Preferred Networks 金融チームが開発している、金融時系列処理のためのPythonフレームワークです。
データ形式の仕様定義、金融時系列データを効率的に扱うためのクラス/関数群、および金融時系列モデルの評価フレームワークが含まれます。

qfeval-functionsは、qfevalの中でも、金融時系列データを効率的に扱うための関数群を提供します。"""

    vocabularies = [
        ("qfeval", "qfeval"),
        ("qfeval-functions", "qfeval-functions"),
        ("qfeval_functions", "qfeval_functions"),
        ("Preferred Networks", "Preferred Networks"),
    ]

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

    client = translate.MCPClient(stream=False)

    # 基本的な翻訳テスト
    input_text = "Hello, world! This is a test."
    outputs = translate_text(
        client, from_lang, to_lang, input_text, context_text, vocabularies
    )
    print("基本翻訳テスト:")
    print(outputs)
    print()

    # .poファイル翻訳を実行
    print("=== .poファイル翻訳開始 ===")
    translate_all_po_files(client)
