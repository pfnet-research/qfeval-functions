import asyncio
import logging
import multiprocessing
import os
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
        messages.append(
            {
                "role": "user",
                "content": f"vocabularies lang={from_lang}|{to_lang} order=shuffled\n{create_vocabulary_format(vocabularies)}",
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

    input_text = "Hello, world! This is a test."
    outputs = translate_text(
        client, from_lang, to_lang, input_text, context_text, vocabularies
    )

    print(outputs)
