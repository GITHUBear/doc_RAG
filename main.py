import logging
from prompt_list import *
from llm_tongyi import TongyiLLM
from http import HTTPStatus
from search_engine import ChromaSearchEngine, MilvusSearchEngine
import gradio as gr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tongyi = TongyiLLM()
engine = MilvusSearchEngine(logger)

def parse_model_output_t1(output):
    logger.info(f"intent detection: {output}")
    # Split the output into lines
    lines = output.strip().split('\n')

    # Initialize the result dictionary
    result = {
        'intent': None,
        'rewritten_question': None
    }
    # Iterate over the lines and extract the relevant information
    for line in lines:
        if line.startswith('意图分类：'):
            result['intent'] = line.split('：')[-1].strip()
            result['intent'] = 1 if result['intent'] == '闲聊' else 0
        elif line.startswith('改写后的问题：'):
            rewritten_question = line.split('：')[-1].strip()
            if rewritten_question != '无':
                result['rewritten_question'] = rewritten_question
            else:
                result['rewritten_question'] = None
    return result

def handle_resp(resp):
    if resp.status_code == HTTPStatus.OK:
        return resp.output.text
    else:
        raise ValueError(f"{resp.code}: {resp.message}")

def format_document_snippet(result, same_doc_idx):
    document_snippet = ""
    for doc_idx in same_doc_idx:
        doc_meta = result[doc_idx[0]]['metadata']
        formatted_string = f"文档路径: {doc_meta['doc_url']}\n"
        formatted_string += f"文档标题: {doc_meta['doc_name']}\n"
        formatted_string += "该文档中可能相关的文档片段:\n"
        for i, section_idx in enumerate(doc_idx):
            section = result[section_idx]
            formatted_string += f"片段 {i+1}:\n片段所在的子标题层级: {section['metadata']['enhanced_title']}\n"
            formatted_string += f"片段内容:\n{section['document']}\n\n"
        document_snippet += formatted_string
    return document_snippet

def get_url_list(result, same_doc_idx):
    url_list = []
    for doc_idx in same_doc_idx:
        doc_meta = result[doc_idx[0]]['metadata']
        url_list.append(doc_meta['doc_name'] + " : " +  doc_meta['doc_url'])
    return url_list

def chat(query: str):
    prompt_1 = prompt_t1.format(user_question = query)
    logger.info(f"############# prompt1:\n {prompt_1}\n")
    resp = tongyi.chat(prompt_1)
    logger.info(f"############# resp1:\n {handle_resp(resp)}\n")
    intent = parse_model_output_t1(handle_resp(resp))['intent']
    resp2 = None
    if intent == 1:
        prompt_2 = prompt_t3.format(user_question = query)
        logger.info(f"############# prompt2:\n {prompt_2}\n")
        resp2 = tongyi.chat(prompt_2)
        logger.info(f"############# resp2:\n {handle_resp(resp2)}\n")
        return handle_resp(resp2)
    elif intent == 0:
        query = parse_model_output_t1(handle_resp(resp))['rewritten_question']
        logger.info(f"############# rewrite query:\n {query}\n")
        results, same_doc_idxs = engine.search([query])
        # for r in results[0]:
        #     print(f"\n\n#############results###############\n  {r['metadata']}")
        document_snippets = format_document_snippet(results[0], same_doc_idxs[0])
        logger.info(f"############# document_snippets:\n {document_snippets}\n")
        prompt_2 = prompt_t2.format(user_question = query, document_snippets = document_snippets)
        ans_f = handle_resp(tongyi.chat(prompt_2))
        ref_str = '结果仅供参考，如有遗漏请参阅下方的参考链接原始文档。 参考链接：\n'
        url_list = '\n'.join(get_url_list(results[0], same_doc_idxs[0]))
        ref_str += url_list
        return ans_f + "\n\n" + ref_str
    else:
        return "执行出错，非常抱歉，请重试或者联系管理员"
    
# chat_resp = chat("OceanBase是什么")
# print("\n\n####################### chat result ###################################\n\n")
# print(chat_resp)

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="OceanBase 问答机器人-V0.1",
    description="基于混合检索和意图分类的OceanBase 问答机器人"
)
demo.launch()