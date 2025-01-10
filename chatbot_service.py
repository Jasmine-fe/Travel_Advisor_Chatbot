from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatbotService:
    def __init__(self, system_message: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.prompt_template = self._create_prompt_template(system_message)
        self.chain = self._create_chain()

    def _create_prompt_template(self, system_message: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{message}")
        ])

    def _create_chain(self):
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.chat_memory.messages
            )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def process_message(self, message: str) -> str:
        response = self.chain.invoke({"message": message})
        self.memory.save_context({"input": message}, {"output": response})
        return response