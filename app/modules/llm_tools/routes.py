from flask import (
    render_template, request, after_this_request, send_file, redirect, url_for, flash
)
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter, TextFormatter, Formatter
from urllib.parse import urlparse, parse_qs

import tiktoken
import langchain
# from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
# from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
# from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
# from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import re
import time
import fitz  # PyMuPDF for rendering first page as PDF
from base64 import b64encode
import requests
from html2image import Html2Image
# from io import BytesIO
import os
import tempfile
import random
import string

MAX_TOKENS=3900
MODEL="gpt-3.5-turbo"
MAX_PAGES=10

# Count number of tokens
def count_tokens(text, model=MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

from . import llm_tools_bp

class MyCustomFormatter(Formatter):
    def format_transcript(self, transcript, **kwargs):
        ft = ''
        for t in transcript:
            # print(f" {t['start']} --> {t['duration']} --> {t['text']}")
            ft = ft + f" {t['start']}: {t['text']} \n"
        return ft

    def format_transcripts(self, transcripts, **kwargs):
        # Do your custom work in here to format a list of transcripts, but return a string.
        return 'your processed output data as a string.'

def stream_log(message, d=0):
    llm_tools_bp.stream_log(" "*d + message)
    
def stream_output(message):
    llm_tools_bp.stream_output(message)
    
def stream_token(message):
    llm_tools_bp.stream_token(message)

import sys
from typing import TYPE_CHECKING, Any, Dict, List
import threading

class MyLLMTokenCallbackHandler(OpenAICallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        """Run when LLM ends running."""
        sys.stdout.write("!!!!!!!!!!!!!!!!on_llm_start\n")
        sys.stdout.flush()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        """Run when LLM ends running."""
        sys.stdout.write("!!!!!!!!!!!!!!!!on_llm_end\n")
        sys.stdout.flush()
        print(response)
        print(kwargs)
        print("-"*50)

class MyChainCallbackHandler(StreamingStdOutCallbackHandler):
    start_streaming: bool = False
    chain_count: int = 0
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.start_streaming:
            # sys.stdout.write(token)
            # sys.stdout.flush()
            stream_output(token)
        else:
            stream_token(f"{token}")
            
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        # kwargs:{'run_id': UUID('c49f8c98-bc88-4d8d-9078-2413a708a269'), 'parent_run_id': None, 'tags': [], 'metadata': {}, 'name': 'MapReduceDocumentsChain'}
        # kwargs:{'run_id': UUID('697d000e-8db2-416c-900b-f15c9321af3d'), 'parent_run_id': UUID('c49f8c98-bc88-4d8d-9078-2413a708a269'), 'tags': [], 'metadata': {}}
        # kwargs:{'run_id': UUID('09d913ef-fa6d-48d1-852b-ab250a50aac3'), 'parent_run_id': UUID('c49f8c98-bc88-4d8d-9078-2413a708a269'), 'tags': [], 'metadata': {}, 'name': 'LLMChain'}
        chain_type = kwargs['name'] if 'name' in kwargs else "Reduce Chain"
        indent = 2 if kwargs['parent_run_id'] == None else 4
        
        if chain_type in ["MapReduceDocumentsChain", "Reduce Chain"]:
            stream_log(f"&gt; {chain_type} started", indent)
        
        if chain_type == "Reduce Chain":
            stream_log(f"document[{self.chain_count}]: ", indent)
            self.chain_count = self.chain_count + 1
        
        if chain_type == "LLMChain":
            stream_log(" ", 4)
            stream_log(f"&gt; {chain_type} started", indent)
            self.start_streaming = True
            stream_log("<y>Start streaming to web client</y>", 4)
            stream_log("Final summary is expected", 4)
        else:
            self.start_streaming = False
            # stream_log("<y>Stop streaming</y>", 4)


@llm_tools_bp.route('/summarize')
def summarize():
    return render_template('llm_tools/summarize.html', default_presets=[])

def create_thumbnail_from_pdf(url):
    """
    Extracts the first page of a PDF file from a web URI.

    Args:
    - url (str): The URL of the PDF file.

    Returns:
    - Page object: The first page of the PDF file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as temp_pdf_file:
        try:
            response = requests.get(url)
            # Write data to the temporary file
            temp_pdf_file.write(response.content)
            temp_pdf_file.flush()
        except Exception as e:
            stream_log(f"<r>Error</r>: Couldn't download the PDF file {e}", 4)
            return [0]
        finally:
            stream_log(f"Generate thumbnail image of the first page", 4)
            document = fitz.open(temp_pdf_file.name, filetype="pdf")
            first_page = document.load_page(0)
            pix = first_page.get_pixmap()
            img_data = pix.tobytes()
            return img_data

@llm_tools_bp.route('/summarize/get-thumbnail-image', methods=['POST'])
def get_thumbnail_image():
    stream_log("Preparing to get thumbnail image")
    json_response = {
        "status": "success",
        "description": "Successfuly completed operation",
    }
    
    input_url = request.form.get("URLInput", "")
    stream_log(f"Check if {input_url} is a URL")
    if not is_url(input_url):
        stream_log("<r>Error</r>: Not a valid URL")
        return {
            "status": "error",
            "description": "Not a valid URL",
        }
    
    stream_log(f"Detecting resource type")
    resource_type = detect_resource_type(input_url)
    if resource_type == "invalid":
        return {
            "status": "error",
            "description": "Not a supported resource type! Provide a valid YouTube video URL, or a valid URL to a webpage with text content or a PDF document",
        }
    
    json_response["resource_type"] = resource_type
    
    if resource_type == "youtube":
        stream_log(f"<g>Found</g> a YouTube video")
        video_url = input_url
        
        stream_log("Extracting video id", 2)
        video_id = extract_video_id(video_url)
        if video_id is None:
            return {
                "status": "error",
                "description": "Although the URL is a valid YouTube video URL, the video id could not be extracted",
            }
        
        stream_log(f"video_id: {video_id}", 2)
        json_response["video_id"] = video_id
        t_uri = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
        stream_log(f"thumbnail_uri: {t_uri}", 2)
        json_response["thumbnail_uri"] = t_uri
    
    if resource_type == "pdf":
        stream_log(f"<g>Found</g> a PDF file")
        stream_log(f"Load and create thumnbnail of the first page", 2)
        img_data = create_thumbnail_from_pdf(input_url)
        base64_encoded = b64encode(img_data).decode("utf-8")
        t_uri = f"data:image/png;base64,{base64_encoded}"
        stream_log(f"thumbnail_uri: {t_uri[:30]}...", 2)
        json_response["thumbnail_uri"] = t_uri
    
    if resource_type == "webpage":
        stream_log(f"<g>Found</g> a Web Resource")
        stream_log(f"Load and create thumnbnail of the first page", 2)
        # t_uri = f"https://s.wordpress.com/mshots/v1/{input_url}"
        tmp_png_path = get_random_string(16)+".png"
        try:
            print(tmp_png_path)
            hti = Html2Image()
            hti.screenshot(url=input_url, save_as=tmp_png_path)
            img_data = []
            with open(tmp_png_path, "rb") as image_file:
                img_data = image_file.read()
            
            base64_encoded = b64encode(img_data).decode("utf-8")
            t_uri = f"data:image/png;base64,{base64_encoded}"
            stream_log(f"thumbnail_uri: {t_uri[:30]}...", 2)
            json_response["thumbnail_uri"] = t_uri
        except Exception as e:
            stream_log(f"<r>Error</r>: Couldn't download the PDF file {e}", 4)
            json_response["thumbnail_uri"] = [0]
        finally:
            os.remove(tmp_png_path)
    
    return json_response

def get_random_string(length):
    """Create random string from all lowercase letter"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str

# tmp_png_path = get_random_string(16)+".png"
# try:
#     print(tmp_png_path)
#     hti = Html2Image()
#     hti.screenshot(url=input_url, save_as=tmp_png_path)
#     img_data = []
#     with open(tmp_png_path, "rb") as image_file:
#         img_data = image_file.read()
#     print(len(img_data))
# except Exception as e:
#     print(f"Error: {e}")

# with tempfile.NamedTemporaryFile(delete=True) as temp_img_file:
#     print(os.path.basename(temp_img_file.name))
#     hti = Html2Image(temp_path=tempfile.gettempdir())
#     hti.screenshot(url=input_url, save_as=os.path.basename(temp_img_file.name))
#     img_data = []
#     with open(temp_img_file.name, "rb") as image_file:
#         img_data = temp_img_file.read()
#     print(len(img_data))

@llm_tools_bp.route('/summarize/get-summarized-text', methods=['POST'])
def get_summarized_text(language='en'):
    start_time = time.time()
    
    stream_log("Preparing to get summarized text")
    json_response = {
        "status": "success",
        "description": "Successfuly completed operation",
    }
    
    input_url = request.form.get("URLInput", "")
    
    stream_log(f"Check if {input_url} is a URL")
    if not is_url(input_url):
        stream_log("<r>Error</r>: Not a valid URL")
        return {
            "status": "error",
            "description": "Not a valid URL",
        }
    
    stream_log(f"Detecting resource type")
    resource_type = detect_resource_type(input_url)
    if resource_type == "invalid":
        return {
            "status": "error",
            "description": "Not a supported resource type! Provide a valid YouTube video URL, or a valid URL to a webpage with text content or a PDF document",
        }
    
    json_response["resource_type"] = resource_type
    docs = []
    
    if resource_type == "youtube":
        stream_log(f"<g>Found</g> a YouTube video")
        video_url = input_url
        
        stream_log("Extracting video id", 2)
        video_id = extract_video_id(video_url)
        if video_id is None:
            return {
                "status": "error",
                "description": "Although the URL is a valid YouTube video URL, the video id could not be extracted",
            }
        
        stream_log(f"video_id: {video_id}", 2)
        json_response["video_id"] = video_id
    
        stream_log(f"Getting transcript for video: {video_id} in language {language}", 2)
        sts, transcript_text = get_transcript(video_id, language)
        if sts == 'E':
            return {
                "status": "error",
                "description": transcript_text,
            }
        
        stream_log(f"<g>Success</g>: Retrieved video transcript", 0)
        # Count tokens and split the text to ensure chunks will fit in context
        stream_log("Counting tokens and splitting the text to ensure chunks will fit in context", 0)
        n_token = count_tokens(transcript_text)
        stream_log(f"Counted {n_token} token(s) while allowed model({MODEL}) context window is {MAX_TOKENS}", 0)
        
        if n_token >= MAX_TOKENS:
            stream_log("<y>Warning</y>: Token count will not fit in context, it will be splitted into chunks", 2)
            splitter=langchain.text_splitter.RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=MODEL)
            chunks=splitter.split_text(transcript_text)
            stream_log(f"Total {str(n_token)} tokens will be splitted into {str(len(chunks))} chunks", 2)
            
            # Create documents from chunks
            stream_log("Create documents from chunks", 4)
            docs = [Document(page_content=chunk) for chunk in chunks] 
            stream_log(f"{str(len(docs))} documents created", 4)            
        else:
            stream_log("Create JUST one document", 2)
            docs = [Document(page_content=transcript_text)] 
            
    if resource_type == "pdf":
        stream_log(f"<g>Found</g> a PDF file")
        stream_log(f"Load and split file", 2)
        loader = PyPDFLoader(input_url)
        t_docs = loader.load_and_split()
        stream_log(f"{str(len(t_docs))} pages found", 2)
        
        stream_log(f"<y>Optimize token count/page for utilizing maximum context window<y>", 2)
        stream_log(f"Stuffing documents", 4)
        # stream_log(f"  Page# |    Char# |   Token#", 4)
        # stream_log(f"-"*30, 4)
        ttc = 0
        ctc = 0
        ct = ""
        for i, doc in enumerate(t_docs):
            # stream_log(f"| {str(i+1).rjust(5)} | {str(len(doc.page_content)).rjust(8)} | {str(count_tokens(doc.page_content)).rjust(8)} |", 4)
            t = count_tokens(doc.page_content)
            ttc = ttc + t # total tokens count
            if ctc + t <= MAX_TOKENS:
                ctc = ctc + t
                ct = ct + "\n" +doc.page_content
            else:
                d = Document(page_content=ct)
                docs.append(d)
                ctc = t
                ct = doc.page_content
        
         # Append any remaining concatenation
        if len(ct) > 0:
            d = Document(page_content=ct)
            docs.append(d)
        
        stream_log(f"<g>{str(len(t_docs))} pages reduced to {str(len(docs))} pages for {MAX_TOKENS} context window<g>", 2)
        # for i, doc in enumerate(docs):
        #     stream_log(f"Page {i+1}: {len(doc.page_content)} -> {count_tokens(doc.page_content)}", 2)
        #     # print(doc.page_content)
        # print(docs[0].page_content)
    
    if resource_type == "webpage":
        stream_log(f"<g>Found</g> a Web Resource")
        stream_log(f"Load and split web page", 2)
        loader = AsyncHtmlLoader(input_url)
        t_docs = loader.load_and_split()
        stream_log(f"{str(len(t_docs))} pages found", 2)
        
        stream_log(f"Scraping text from html", 2)
        html2text = Html2TextTransformer()
        t_docs = html2text.transform_documents(t_docs)
        
        stream_log(f"<y>Optimize token count/page for utilizing maximum context window<y>", 2)
        stream_log(f"Stuffing documents", 4)
        # stream_log(f"  Page# |    Char# |   Token#", 4)
        # stream_log(f"-"*30, 4)
        ttc = 0
        ctc = 0
        ct = ""
        for i, doc in enumerate(t_docs):
            # stream_log(f"| {str(i+1).rjust(5)} | {str(len(doc.page_content)).rjust(8)} | {str(count_tokens(doc.page_content)).rjust(8)} |", 4)
            t = count_tokens(doc.page_content)
            ttc = ttc + t # total tokens count
            if ctc + t <= MAX_TOKENS:
                ctc = ctc + t
                ct = ct + "\n" +doc.page_content
            else:
                d = Document(page_content=ct)
                docs.append(d)
                ctc = t
                ct = doc.page_content
        
         # Append any remaining concatenation
        if len(ct) > 0:
            d = Document(page_content=ct)
            docs.append(d)
        
        stream_log(f"<g>{str(len(t_docs))} pages reduced to {str(len(docs))} pages for {MAX_TOKENS} context window<g>", 2)
        for i, doc in enumerate(docs):
            stream_log(f"Page {i+1}: {len(doc.page_content)} -> {count_tokens(doc.page_content)}", 2)
            # print(doc.page_content)
        print(docs[0].page_content)
        
    
    summary = ""
    
    stream_log("Creating an LLM instance", 0)
    temperature=0
    stream_log(f"temperature: {temperature}", 2)
    stream_log(f"model_name: {MODEL}", 2)
    stream_log("Attaching OpenAICallbackHandler to track token stats", 2)
    lcb = MyLLMTokenCallbackHandler()
    llm = ChatOpenAI (
        api_key=os.environ.get('OAK'),
        temperature=temperature, 
        model_name=MODEL, 
        streaming=True,
        callbacks=[lcb]
    )
        
    # Create a map reduce chain from chunks
    stream_log("Creating a map reduce chain", 0)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, 
            #  callbacks = [MyCustomCallbackHandler()]
    )
    # chain.callbacks = [MyCustomCallbackHandler()]
    if len(docs) > MAX_PAGES:
        stream_log(f"<r>Limiting the number of pages to {MAX_PAGES}</r>", 0)
            
    stream_log("Running the chain", 0)
    stream_log("Attaching MyCustomCallbackHandler to stream results to client over socket.io", 2)
    stream_log("Attaching MyCustomCallbackHandler to track token stats", 2)
    ccb = MyChainCallbackHandler()
    summary = chain.run(docs[:MAX_PAGES], callbacks=[ccb])
    
    
    stream_log(f"<g>Completed</g>: {summary}", 0)
    json_response["summary"] = summary

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    stream_log(f"<g>Execution time: {execution_time} seconds</g>")
    
    json_response["execution_time"] = execution_time
    return json_response

def is_url(url):
    try:
        result = urlparse(url)
        # <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        # return all([result.scheme, result.netloc])
        stream_log("Check id scheme is http or https", 2)
        return result.scheme.lower() in ['http', 'https']
    except ValueError:
        return False
    
def detect_resource_type(url):
    try:
        result = urlparse(url)
        # <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        # return all([result.scheme, result.netloc])
        netloc = result.netloc.lower()
        path = result.path.lower()
        stream_log(f"Extracted netloc: {netloc}", 2)
        stream_log(f"Extracted path: {path}", 2)
        # Regular expression pattern for YouTube video URLs
        youtube_pattern = re.compile(
            r'(https?://)?(www\.)?'
            r'(youtube\.com/watch\?v=|youtu\.be/)'
            r'(?P<video_id>[A-Za-z0-9_-]{11})'
        )
        # Check if the URL matches the pattern
        match = youtube_pattern.match(url)
        if match:
            stream_log(f"This is a YouTube video link", 2)
            return "youtube"
        
        # Split the path by the dot (.) character and get the last part (file extension)
        file_extension = path.split('.')[-1].lower()
        if file_extension == 'pdf':
            stream_log(f"This is a PDF file link", 2)
            return "pdf"
        
        # Treat rest as a webpage
        return "webpage"
        
    except ValueError:
        stream_log("<r>Error</r>: Not a valid URL", 2)
        return "invalid"
    
def extract_video_id(url):
    # Regular expression pattern for YouTube video URLs
    youtube_pattern = re.compile(
        r'(https?://)?(www\.)?'
        r'(youtube\.com/watch\?v=|youtu\.be/)'
        r'(?P<video_id>[A-Za-z0-9_-]{11})'
    )
    # Check if the URL matches the pattern
    match = youtube_pattern.match(url)
    if match:
        return match.group('video_id')
    else:
        stream_log("<r>Error</r>: Couldn't find video id", 2)
        return None

def get_transcript(video_id, language='en'):
    transcript_text = ""
    # Find transcript in the specified language
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(language)
        transcript_text = transcript.fetch()
        transcript_text = TextFormatter().format_transcript(transcript_text)
        stream_log(f"Transcript found in {language} for video {video_id}", 2)
        return ('S', transcript_text)
    except NoTranscriptFound as e: 
        stream_log(f"<r>Error</r>: Transcript not found in {language}, searching auto generated ", 2)
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            transcript_text = transcript.fetch()
            # print(transcript_text)
            transcript_text = TextFormatter().format_transcript(transcript_text)
            stream_log(f"Auto generated transcript found in {language} for video {video_id}", 2)
            return ('S', transcript_text)
        except NoTranscriptFound as e: 
            # TODO: try translating!
            stream_log(f"<r>Error</r>: No transcript found in {language} for video {video_id}", 2)
            stream_log("<y>TODO<y>: Consider implementing on-demand translation", 2)
            return ('E', f"No Transcript Found in {language} for video {video_id}")
    except Exception as e:
        stream_log(f"<r>Error</r>: An error occurred while fetching the transcript", 2)
        stream_log(f"{str(e)}", 2)
        return ('E', f"An error occurred while fetching the transcript: {str(e)}")

@llm_tools_bp.route('/image')
def image():
    return render_template('llm_tools/image.html', default_presets=[])

# https://s.wordpress.com/mshots/v1/https://lilianweng.github.io/posts/2023-06-23-agent/
# https://s.wordpress.com/mshots/v1/https://python.langchain.com/docs/integrations/document_transformers/html2text
# https://s.wordpress.com/mshots/v1/https://www.youtube.com/watch?v=y2cRcOPHL_U


# https://s.wordpress.com/mshots/v1/https://scikit-learn.org/stable/auto_examples/index.html
