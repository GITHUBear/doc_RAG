import time
import logging
from prompt_list import *
from llm_tongyi import TongyiLLM
from llm_zhipu import ZhipuLLM
from search_engine import MilvusSearchEngine

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

tongyi = TongyiLLM()
llm_model = ZhipuLLM()
engine = MilvusSearchEngine(logger)

def parse_model_output_t1(output):
    logger.debug(f"intent detection: {output}")
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
    start_time = time.time()
    prompt_1 = prompt_t1.format(user_question = query)
    logger.debug(f"############# prompt1:\n {prompt_1}\n")
    resp = llm_model.chat(prompt_1)
    intent_chat_end_time = time.time()
    logger.debug(f"############# resp1:\n {resp}\n")
    intent = parse_model_output_t1(resp)['intent']
    resp2 = None
    if intent == 1:
        prompt_2 = prompt_t3.format(user_question = query)
        logger.debug(f"############# prompt2:\n {prompt_2}\n")
        resp2 = llm_model.chat(prompt_2)
        simple_chat_end_time = time.time()
        logger.debug(f"############# resp2:\n {resp2}\n")
        logger.info(f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms  simple_chat:{(simple_chat_end_time - intent_chat_end_time) * 1000}ms")
        return resp2
    elif intent == 0:
        query = parse_model_output_t1(resp)['rewritten_question']
        logger.debug(f"############# rewrite query:\n {query}\n")
        results, same_doc_idxs = engine.search([query])
        document_snippets = format_document_snippet(results[0], same_doc_idxs[0])
        doc_search_end_time = time.time()
        logger.debug(f"############# document_snippets:\n {document_snippets}\n")
        prompt_2 = prompt_t2.format(user_question = query, document_snippets = document_snippets)
        ans_f = llm_model.chat(prompt_2)
        rag_end_time = time.time()
        logger.info(f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms doc_search:{(doc_search_end_time - intent_chat_end_time) * 1000}ms rag_chat:{(rag_end_time - doc_search_end_time) * 1000}ms")
        ref_str = '结果仅供参考，如有遗漏请参阅下方的参考链接原始文档。 参考链接：\n'
        url_list = '\n'.join(get_url_list(results[0], same_doc_idxs[0]))
        ref_str += url_list
        return ans_f + "\n\n" + ref_str
    else:
        return "执行出错，非常抱歉，请重试或者联系管理员"

def handle_multi_chat_resp(ret_code, resp):
    if ret_code == 200:
        return resp
    else:
        return f"get http error: {ret_code}"

history_msgs = []
def multi_chat(query: str):
    global history_msgs
    start_time = time.time()
    prompt_1 = prompt_t0.format(chat_context = history_msgs, user_question = query)
    logger.debug(f"############# prompt1:\n {prompt_1}\n")
    resp = tongyi.chat(prompt_1)
    intent_chat_end_time = time.time()
    logger.debug(f"############# resp1:\n {resp}\n")
    intent = parse_model_output_t1(resp)['intent']
    resp2 = None
    if intent == 1:
        prompt_2 = prompt_t3.format(user_question = query)
        logger.debug(f"############# prompt2:\n {prompt_2}\n")
        ret_code, resp2, new_history_msgs = tongyi.multi_chat(history_msgs, prompt_2, query)
        simple_chat_end_time = time.time()
        logger.debug(f"############# resp2:\n {resp2}\n")
        logger.info(f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms  simple_chat:{(simple_chat_end_time - intent_chat_end_time) * 1000}ms")
        history_msgs = new_history_msgs
        return handle_multi_chat_resp(ret_code, resp2)
    elif intent == 0:
        query = parse_model_output_t1(resp)['rewritten_question']
        logger.debug(f"############# rewrite query:\n {query}\n")
        results, same_doc_idxs = engine.search([query])
        document_snippets = format_document_snippet(results[0], same_doc_idxs[0])
        doc_search_end_time = time.time()
        logger.debug(f"############# document_snippets:\n {document_snippets}\n")
        prompt_2 = prompt_t2.format(user_question = query, document_snippets = document_snippets)
        ret_code, resp2, new_history_msgs = tongyi.multi_chat(history_msgs, prompt_2, query)
        rag_end_time = time.time()
        logger.info(f"cost --- intent_chat:{(intent_chat_end_time - start_time) * 1000}ms doc_search:{(doc_search_end_time - intent_chat_end_time) * 1000}ms rag_chat:{(rag_end_time - doc_search_end_time) * 1000}ms")
        ref_str = '结果仅供参考，如有遗漏请参阅下方的参考链接原始文档。 参考链接：\n'
        url_list = '\n'.join(get_url_list(results[0], same_doc_idxs[0]))
        ref_str += url_list
        ans_f = handle_multi_chat_resp(ret_code, resp2) 
        history_msgs = new_history_msgs
        if ret_code == 200:
            return ans_f + "\n\n" + ref_str
        else:
            return ans_f
    else:
        return "执行出错，非常抱歉，请重试或者联系管理员"

def multi_chat_not_restrict(query: str):
    global history_msgs
    if len(history_msgs) == 0:
        return multi_chat(query)
    else:
        start_time = time.time()
        ret_code, resp2, new_history_msgs = tongyi.multi_chat(history_msgs, query, query)
        simple_chat_end_time = time.time()
        logger.info(f"cost --- simple_chat:{(simple_chat_end_time - start_time) * 1000}ms")
        history_msgs = new_history_msgs
        return handle_multi_chat_resp(ret_code, resp2)
    
# chat_resp = chat("OceanBase是什么")
# print("\n\n####################### chat result ###################################\n\n")
# print(chat_resp)
