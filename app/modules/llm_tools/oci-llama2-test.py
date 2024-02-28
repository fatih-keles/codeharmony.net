import os
os.environ['OCI_CLI_SUPPRESS_FILE_PERMISSIONS_WARNING'] = 'true'



config = {
    "user": os.environ.get('OCI_USER'),
    "key_content": "<content_of_the_private_key_not_in_key_file>",
    "fingerprint": os.environ.get('OCI_FINGERPRINT'),
    "tenancy": os.environ.get('OCI_TENANCY'),
    "region": os.environ.get('OCI_REGION'),
}




# from oci.config import validate_config
# validate_config(config)

# from oci.config import from_file
# config = from_file()
# config['region'] = 'us-chicago-1'
# print(config)

# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# generate_text_demo.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model for given prompts using OCI Generative AI Service.
##########################################################################
# Application Command line(no parameter needed)
# python generate_text_demo.py
##########################################################################
import oci

# Setup basic variables
# Auth Config
# TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
compartment_id = os.environ.get('OCI_TENANCY')
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
config['region'] = 'us-chicago-1'
print(config)

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
llm_inference_request = oci.generative_ai_inference.models.LlamaLlmInferenceRequest()
llm_inference_request.prompt = """
ROLE: You are a RESPECTFUL and RESPONSIBLE comedian. 
TASK: You are tasked with creating a joke about Olympic Games suitable for general in a light-hearted manner. 
You must adjust the joke to fit teenagers and consider gender neutral perspectives if applicable.
Reply only with the joke, do not explain.
JOKE:
"""

llm_inference_request.max_tokens = 600
llm_inference_request.temperature = 0.9
llm_inference_request.frequency_penalty = 1
llm_inference_request.top_p = 0.75

# generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyai3pxxkeezogygojnayizqu3bgslgcn6yiqvmyu3w75ma")
generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="meta.llama-2-70b-chat")
generate_text_detail.inference_request = llm_inference_request
generate_text_detail.compartment_id = compartment_id
generate_text_response = generative_ai_inference_client.generate_text(generate_text_detail)
# Print result
print("**************************Generate Texts Result**************************")
print(generate_text_response.data)

# %%
!pip install pypdf

# %%
from langchain.document_loaders import PyPDFLoader
input_file = "https://www.oracle.com/a/ocom/docs/paas-iaas-universal-credits-3940775.pdf"
loader = PyPDFLoader(input_file)
pages = loader.load_and_split()