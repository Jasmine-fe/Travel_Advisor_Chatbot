from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatbotService:
    def __init__(self, system_message: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompt_template = self._create_prompt_template(system_message)
        self.chain = self._create_chain()

    def _create_prompt_template(self, system_message: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{message}")
        ])

    def _create_chain(self):
        return (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def process_message(self, message: str) -> str:
        response = self.chain.invoke({"message": message})
        return response