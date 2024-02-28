import os
import oci

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_oci import OCIGenAI

print(os.environ.get('OCI_TENANCY'))

llm = OCIGenAI(
    auth_profile='GENAI',
    model_id="cohere.command",
    service_endpoint="https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com",
    compartment_id=os.environ.get('OCI_TENANCY')
)

config = oci.config.from_file('~/.oci/config', 'GENAI')

response = llm("Tell me a joke.", temperature=0)
print(response)
prompt = PromptTemplate(input_variables=["query"], template="{query}")
llm_chain = LLMChain(llm=llm, prompt=prompt)
response= llm_chain("what is the capital of france")
print(response)