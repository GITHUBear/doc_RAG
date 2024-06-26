import redis
import json
from config import RagConfig

class ChatSession:
    def __init__(self):
        config = RagConfig()
        self.redis_conn = redis.StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
        self.ttl = config.redis_ttl

    def get_session(self, conversation_id: str, nickname: str):
        session_id = conversation_id + nickname
        session = self.redis_conn.get(session_id)
        if not session:
            return {
                'status': 1,   # done
                'conversation_id': conversation_id,
                'nickname': nickname,
                'messages': [],
            }
        return json.loads(session.decode('utf-8'))
    
    def update_session(self, conversation_id: str, nickname: str, session: dict):
        session_id = conversation_id + nickname
        self.redis_conn.set(session_id, json.dumps(session))
        self.redis_conn.expire(session_id, self.ttl)