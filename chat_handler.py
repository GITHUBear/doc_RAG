from chat_session import ChatSession
from llm_tongyi import TongyiLLM

def handle_chat(conversation_id: str, nickname: str, user_message: str):
    chat_session = ChatSession()
    session = chat_session.get_session(conversation_id, nickname)
    if session['status'] == 0:
        return "操作频繁，稍后再试"
    
    # tongyi = TongyiLLM()
    # return_code, resp, messages = tongyi.multi_chat(session['messages'], user_message)
    # session[]
    # return resp