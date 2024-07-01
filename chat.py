import os
import time
import logging
from prompt_list import *
from llm_abc import LLM
from llm_tongyi import TongyiLLM
from llm_zhipu import ZhipuLLM
from search_engine import MilvusSearchEngine
from typing import List, Dict, Any, Generator
from message import StreamChunk

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(
    logging.DEBUG if os.environ.get("DEBUG", "false") == "true" else logging.INFO
)

tongyi = TongyiLLM()
zhipu = ZhipuLLM()

model_map: Dict[str, LLM] = {"tongyi": tongyi, "zhipu": zhipu}

engine = MilvusSearchEngine(logger)


def intention_detection(output) -> Dict[str, Any]:
    logger.debug(f"intent detection: {output}")
    # Split the output into lines
    lines = output.strip().split("\n")

    # Initialize the result dictionary
    result = {"intent": None, "rewritten_question": None}
    # Iterate over the lines and extract the relevant information
    for line in lines:
        if line.startswith("意图分类："):
            result["intent"] = line.split("：")[-1].strip()
            result["intent"] = 1 if result["intent"] == "闲聊" else 0
        elif line.startswith("改写后的问题："):
            rewritten_question = line.split("：")[-1].strip()
            if rewritten_question != "无":
                result["rewritten_question"] = rewritten_question
            else:
                result["rewritten_question"] = None
    return result


def format_document_snippet(result, same_doc_idx):
    document_snippet = ""
    for doc_idx in same_doc_idx:
        doc_meta = result[doc_idx[0]]["metadata"]
        formatted_string = f"文档路径: {doc_meta['doc_url']}\n"
        formatted_string += f"文档标题: {doc_meta['doc_name']}\n"
        formatted_string += "该文档中可能相关的文档片段:\n"
        for i, section_idx in enumerate(doc_idx):
            section = result[section_idx]
            formatted_string += f"片段 {i+1}:\n片段所在的子标题层级: {section['metadata']['enhanced_title']}\n"
            formatted_string += f"片段内容:\n{section['document']}\n\n"
        document_snippet += formatted_string
    return document_snippet


def get_url_list(
    result,
    same_doc_idx,
    output_md=True,
    replace_from="./oceanbase-doc",
    replace_to="https://github.com/oceanbase/oceanbase-doc/blob/V4.1.0",
) -> List[str]:
    url_list = []
    for doc_idx in same_doc_idx:
        doc_meta = result[doc_idx[0]]["metadata"]
        if output_md:
            url_list.append(
                f"\n- [{doc_meta['doc_name']}]({doc_meta['doc_url'].replace(replace_from, replace_to)})"
            )
        else:
            url_list.append(
                doc_meta["doc_name"]
                + " : "
                + {doc_meta["doc_url"].replace(replace_from, replace_to)}
            )
    return url_list


def chat(query: str, model="tongyi"):
    llm = model_map.get(model, tongyi)
    start_time = time.time()
    prompt_1 = prompt_t1.format(user_question=query)
    logger.debug(f"############# prompt1:\n {prompt_1}\n")
    resp = llm.chat(prompt_1)
    intent_chat_end_time = time.time()
    logger.debug(f"############# resp1:\n {resp}\n")
    intent = intention_detection(resp)["intent"]
    resp2 = None
    if intent == 1:
        prompt_2 = prompt_t3.format(user_question=query)
        logger.debug(f"############# prompt2:\n {prompt_2}\n")
        resp2 = llm.chat(prompt_2)
        simple_chat_end_time = time.time()
        logger.debug(f"############# resp2:\n {resp2}\n")
        logger.info(
            f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms  simple_chat:{(simple_chat_end_time - intent_chat_end_time) * 1000}ms"
        )
        return resp2
    elif intent == 0:
        query = intention_detection(resp)["rewritten_question"]
        logger.debug(f"############# rewrite query:\n {query}\n")
        results, same_doc_idxs = engine.search([query])
        document_snippets = format_document_snippet(results[0], same_doc_idxs[0])
        doc_search_end_time = time.time()
        logger.debug(f"############# document_snippets:\n {document_snippets}\n")
        prompt_2 = prompt_t2.format(
            user_question=query, document_snippets=document_snippets
        )
        ans_f = llm.chat(prompt_2)
        rag_end_time = time.time()
        logger.info(
            f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms doc_search:{(doc_search_end_time - intent_chat_end_time) * 1000}ms rag_chat:{(rag_end_time - doc_search_end_time) * 1000}ms"
        )
        ref_str = "结果仅供参考，如有遗漏请参阅下方的参考链接原始文档。 参考链接：\n"
        url_list = "\n".join(get_url_list(results[0], same_doc_idxs[0]))
        ref_str += url_list
        return ans_f + "\n\n" + ref_str
    else:
        return "执行出错，非常抱歉，请重试或者联系管理员"


def multi_chat(
    query: str,
    history_msgs: List[dict],
    model="tongyi",
    stream=False,
    context_length=4,
) -> str | Generator[bytes, Any, None]:
    llm = model_map.get(model, zhipu)

    pruned_history = history_msgs[-context_length:]

    start_time = time.time()
    intention_classification = prompt_t0.format(
        chat_context=pruned_history, user_question=query
    )
    logger.debug(f"############# prompt1:\n {intention_classification}\n")
    resp = llm.chat(intention_classification)
    intent_chat_end_time = time.time()
    logger.debug(f"############# resp1:\n {resp}\n")
    intent = intention_detection(resp)["intent"]

    if intent == 1:
        polished_query = prompt_t3.format(user_question=query)
        logger.debug(f"############# prompt2:\n {polished_query}\n")
        new_history = pruned_history + [{"role": "user", "content": polished_query}]
        return llm.chat_with_history(messages=new_history, stream=stream)

    elif intent == 0:
        query = intention_detection(resp)["rewritten_question"]

        logger.debug(f"############# rewrite query:\n {query}\n")

        results, same_doc_idxs = engine.search([query])
        document_snippets = format_document_snippet(results[0], same_doc_idxs[0])

        doc_search_end_time = time.time()
        logger.debug(f"############# document_snippets:\n {document_snippets}\n")
        polished_query = prompt_t2.format(
            user_question=query, document_snippets=document_snippets
        )
        new_history = pruned_history + [{"role": "user", "content": polished_query}]

        ref_tips = "结果仅供参考，如有遗漏请参阅下方的参考链接原始文档。 参考链接：\n"

        if not stream:
            llm_resp = llm.chat_with_history(messages=new_history, stream=False)
            rag_end_time = time.time()
            logger.info(
                f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms doc_search:{(doc_search_end_time - intent_chat_end_time) * 1000}ms rag_chat:{(rag_end_time - doc_search_end_time) * 1000}ms"
            )

            url_list = "\n".join(get_url_list(results[0], same_doc_idxs[0]))
            ref_str = ref_tips + url_list

            return llm_resp + "\n\n" + ref_str
        else:
            url_list = get_url_list(results[0], same_doc_idxs[0])
            len_url_list = len(url_list)

            def stream_chat():
                for chunk in llm.chat_with_history(
                    messages=new_history, stream=True, need_append=True
                ):
                    yield chunk

                yield StreamChunk(model=model, content=ref_tips).to_json().encode()

                for i, link in enumerate(url_list):
                    stream_chunk = StreamChunk(
                        model=model, content=link, done=(i == (len_url_list - 1))
                    )
                    yield stream_chunk.to_json().encode()

            return stream_chat()

    else:
        return "执行出错，非常抱歉，请重试或者联系管理员"


def multi_chat_not_restrict(query: str, history_msgs: List[dict]) -> str:
    if len(history_msgs) == 0:
        return multi_chat(query)
    else:
        new_history = history_msgs + [{"role": "user", "content": query}]
        start_time = time.time()
        resp = tongyi.chat_with_history(new_history)
        simple_chat_end_time = time.time()
        logger.info(
            f"cost --- simple_chat:{(simple_chat_end_time - start_time) * 1000}ms"
        )
        return resp


# chat_resp = chat("OceanBase是什么")
# print("\n\n####################### chat result ###################################\n\n")
# print(chat_resp)
