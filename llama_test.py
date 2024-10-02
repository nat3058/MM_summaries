# import llama_index
# from llama_index.core.response_synthesizers import TreeSummarize


# from typing import Optional, List, Mapping, Any

# from llama_index.core import SimpleDirectoryReader, SummaryIndex
# from llama_index.core.callbacks import CallbackManager
# from llama_index.core.llms import (
#     CustomLLM,
#     CompletionResponse,
#     CompletionResponseGen,
#     LLMMetadata,
# )
# from llama_index.core.llms.callbacks import llm_completion_callback
# from llama_index.core import Settings
# # from meta_ai_api import MetaAI
# # from llama_index.llms import MetaAI
# import os
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# from llama_index.core.base.llms import LLM
# from llama_index import ServiceContext

# # llm = MetaAI()
# Settings.llm = MetaAI()

# service_context = ServiceContext.from_defaults(llm=llm,embed_model="local")
# import llama_index.llms
# print(dir(llama_index.core.base.llms.base))
# os.environ["META_AI_API_KEY"] = "<your-api-key>"
# os.environ["META_AI_ENDPOINT"] = "https://www.meta.ai/"

# llm = MetaAI(
#     model="meta-llm-model",
#     api_key=os.environ["META_AI_API_KEY"],
#     endpoint=os.environ["https://www.meta.ai/"]
# )
# response = llm.complete("The sky is a beautiful blue and")
# print(response)

# proxy = {
#     'http': 'localhost:9001',
#     # 'https': 'https://proxy_address:port'
# }

# # ai = MetaAI(proxy=proxy)

# from llama_index.llms.openai_like import OpenAILike
# llm = OpenAILike(
#     model='meta-llama/Llama-2-13b-chat-hf',
#     api_key="my api_key", 
#     # api_base="https://www.meta.ai/"
#     # api_base="https://huggingface.co/chat/"
#     api_base="localhost:9001"
# )



# response = llm.complete("To infinity, and")


from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings

from meta_ai_api import MetaAI
# from pydantic import ClassVar
import time

meta_list = []
for i in range(15):
    meta = MetaAI()
    meta_list.append(meta)

class OurLLM_meta_list(CustomLLM):
    # ai : ClassVar[MetaAI] = MetaAI()
    context_window: int = 4096
    num_output: int = 3900
    model_name: str = "custom"
    num_prom : int = 0
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # resp = meta.prompt(prompt)
        resp = meta_list[self.num_prom % 15].prompt(prompt)
        print(resp["message"])
        self.num_prom = self.num_prom + 1
        print(self.num_prom)
        return CompletionResponse(text=resp["message"])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM_meta_list()

completion_response = Settings.llm.complete("To infinity, and")
print(completion_response)
print("DONE 1")

from llama_index.core.response_synthesizers import TreeSummarize, SimpleSummarize
summarizer = TreeSummarize(use_async=True)

query_str = "Summarize this text"
text_chunks = ["text1", "text2"]

from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api.formatters import JSONFormatter
import json
from textwrap import wrap


def get_video_ids(playlist_url):
    """
    Retrieves video IDs from a public YouTube playlist.

    Args:
        playlist_url (str): URL of the YouTube playlist.
    """

    # Create a Playlist object
    p = Playlist(playlist_url)

    # Get video URLs
    video_urls = p.video_urls

    # Extract video IDs from URLs
    video_ids = [url.split('watch?v=')[-1] for url in video_urls]

    # Print video IDs
    return(video_ids)


video_ids = get_video_ids("https://www.youtube.com/playlist?list=PLVbP054jv0KoZTJ1dUe3igU7K-wUcQsCI")
video_id = video_ids[1]
transcript = YouTubeTranscriptApi.get_transcript(video_id)


print("\n * Formatting into chunks...")
# lets do text formatting
formatter = TextFormatter()
txt_formatted = formatter.format_transcript(transcript)
chnks = wrap(txt_formatted, width=1000)

#     install_num = 1;
#     for c in chnks:
#         installment_req = 'Here is installment number '
#         installment_req += str(install_num)
#         installment_req += ': \n'
#         installment_req += c
#         print(installment_req)
#         print(meta.prompt(installment_req))
#         install_num = install_num + 1


# summary = summarizer.get_response(query_str, chnks)

# from llama_index.core.callbacks import PrintCallbackManager


class CustomCallbackManager(CallbackManager):
    def on_progress(self, progress: float):
        print(f"Summarization progress: {progress*100}%")

    def on_stage(self, stage: str):
        print(f"Summarization stage: {stage}")

    def on_error(self, error: Exception):
        print(f"Summarization error: {error}")


callback_manager = CustomCallbackManager()
'''
summarizer = TreeSummarize(
    use_async=True,
    verbose=True,
    callback_manager=callback_manager,
    llm = Settings.llm



    # num_summarization_levels=3,
    # prompt_template="Summarize the key points of [TEXT].",
    # # summary_length=150,
    # language_model="LLaMA-2",
    # clustering_algorithm="hierarchical",
    # min_chunk_size=200,
    # overlap=0.7,
    # stopwords=["a", "an", "the"]
)
# summary = summarizer.get_response(query_str, chnks)
# print(summary)

import asyncio
async def test_async():
    response = await summarizer.aget_response(query_str, chnks)
    print(response)

asyncio.run(test_async())
'''

''' ----------------------------------------------------------------------'''

'''

one_meta = MetaAI()
class OurLLM_one_meta(CustomLLM):
    # ai : ClassVar[MetaAI] = MetaAI()
    context_window: int = 4090
    num_output: int = 3000
    model_name: str = "custom"
    num_prom : int = 0
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # resp = meta.prompt(prompt)
        resp = one_meta.prompt(prompt)
        print(resp["message"])
        self.num_prom = self.num_prom + 1
        print(self.num_prom)
        return CompletionResponse(text=resp["message"])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM_one_meta()
# summarizer = SimpleSummarize(
#     llm = Settings.llm
# )

# summary = summarizer.get_response(query_str, chnks)
# print("HERE IS SUM:")
# print(summary)



import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

transcript_directory = "./raw_transcripts"
storage_directory = "storage/ancient-aliens-official/local"

# Add filename as metadata to each chunk associated with a document/transcript
# filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
documents = SimpleDirectoryReader(transcript_directory, filename_as_id=True).load_data()

# Exclude metadata from the LLM, meaning it won't read it when generating a response.
# Future - consider looping over documents and setting the id_ to basename, instead of fullpath
# [document.excluded_llm_metadata_keys.append('episode_title') for document in documents]

# chunk_size - It defines the size of the chunks (or nodes) that documents are broken into when they are indexed by LlamaIndex
# service_context = ServiceContext.from_defaults(llm=Settings.llm, chunk_size=1024,
#                                                callback_manager=callback_manager)

# Build the index

import llama_index.embeddings.huggingface
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model= HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(documents,llm=Settings.llm, show_progress=True)

# Persist the index to disk
# index.storage_context.persist(persist_dir=storage_directory)

# from IPython.display import Markdown, display
# from llama_index.prompts import PromptTemplate

#query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=15,response_mode="tree_summarize")
query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=15,response_mode="accumulate")
# Settings.num_output = 512
# Settings.context_window = 4096


p1 = "Please give me a summary of the first quarter of the document with specific insightful bullet points and good headings"
p2 = "Please give me a summary with the key insights or info from the first quarter of the document"
p3 = "Please give me a detailed summary with key info relevant to a retail investor"
p4 = "Please give me a summary of the entire show. Each major segment of the show should have a section with a clear heading and 5 specific insightful subbullet points"


# response = query_engine.query("Please give me a very detailed summary")

response = query_engine.query(p4)
print(response)
'''

# display(Markdown(f"<b>{response}</b>"))








from llama_index.core import SimpleDirectoryReader

transcript_directory = "./raw_transcripts"
reader = SimpleDirectoryReader(transcript_directory)
documents = reader.load_data()



one_meta = MetaAI()
class OurLLM_one_meta(CustomLLM):
    # ai : ClassVar[MetaAI] = MetaAI()
    context_window: int = 4090
    num_output: int = 3000
    model_name: str = "custom"
    num_prom : int = 0
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # resp = meta.prompt(prompt)
        resp = one_meta.prompt(prompt)
        print(resp["message"])
        self.num_prom = self.num_prom + 1
        print(self.num_prom)
        return CompletionResponse(text=resp["message"])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
Settings.llm = OurLLM_one_meta()

from llama_index.core.composability import QASummaryQueryEngineBuilder
import llama_index.embeddings.huggingface
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model= HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



# NOTE: can also specify an existing docstore, summary text, qa_text, etc.
query_engine_builder = QASummaryQueryEngineBuilder(
    llm=Settings.llm ,
)
query_engine = query_engine_builder.build_from_documents(documents)


p1 = "Please give me a summary of the first quarter of the document with specific insightful bullet points and good headings"
p2 = "Please give me a summary with the key insights or info from the first quarter of the document"
p3 = "Please give me a detailed summary with key info relevant to a retail investor"
p4 = "Please give me a summary of the entire show. Each major segment of the show should have a section with a clear heading and 5 specific insightful useful subbullet points"
p5 = "Please give me a summary of the entire show. Each major segment of the show should have a section with a clear heading and 4 specific insightful subbullet points. Keep it under 3000 words total please"
p6 = '''
Please give me a summary of the entire show such that the summary recaps only specfic useful insights 
from each segment of the show. Each major segment of the show should have a section with a clear heading 
and 3 to 5 specific insightful subbullet points for that segment. Avoid adding too vague statements in the summary.
Keep it under 3000 words total please
'''

p7 = '''
Pretend you must give me a summary of the entire show such that the summary recaps only specfic useful insights 
from each unqiue and important segments of the show. Select segments that are unique and contain helpful 
information for an investor.
in the summary, each of those segments should have a section with a clear heading and 
4 to 5 specific insightful subbullet points for that segment.
Don't be vague and make sure you are detailed but still succiont and to the point.
Keep it under 3000 words total please.
'''

p8 = '''Please give me a summary of the entire show such that the summary recaps only specfic useful insights 
from the show. Select 5 to 7 major segments that are unique and contain helpful information for an investor. a
Each selected segment should have a section with a clear heading 
and at least 5 specific insightful useful subbullet points for that segment. '''

'''Select segments that are unique and contain helpful information for an investor'''
# response = query_engine.query(
#     "Can you give me a summary of the show?",
# )
p9='''I am an data-driven knowledge-hungry stock market investor but did not watch Jim Cramer's show yet. 
Quickly, please give me a succint but specific recap of the entire show in the same order 
in which the info/segment was presented.
Make sure that information is grouped by segments directly from the show, 
and each segment has a clear header with detailed concise sub bullet points. 
Don't add information that isn't really helpful or useful to know as a investor.
Do not make it more than 2800 words. Thanks'''
response = query_engine.query(p9)
print("FINAL RESP")
print(response)

'''what was Cramers specfic stock advice for each stock in the lightnign round?'''

while True:
    user_input = input("Please enter something (type 'q' to exit): ")
    if user_input.lower() == "q":
        break
    # Perform your desired action with the user_input
    print(f"You entered: {user_input}")
    meta_op = query_engine.query(user_input)
    #print(meta_op)
    #print(meta_op["message"])
    print(meta_op)
    # with open('transcript_summary.txt', 'w') as txt_file:
    #     txt_file.write(meta_op["message"])
    # with open('transcript_summary_html.html', 'w') as txt_file:
    #     txt_file.write(meta_op["message"])


# define embed model
# Settings.embed_model = "local:BAAI/bge-base-en-v1.5"


# # Load the your data
# documents = SimpleDirectoryReader("./data").load_data()
# index = SummaryIndex.from_documents(documents)

# # Query and print response
# query_engine = index.as_query_engine()
# response = query_engine.query("<query_text>")
# print(response)


# print(response)


# remotely_run_anon = HuggingFaceInferenceAPI(
#     model_name="meta-llama/Meta-Llama-3-70B-Instruct"
# )
# completion_response = remotely_run_anon.complete("To infinity, and")
# print(completion_response)
# print("DONE 1")



# summarizer = TreeSummarize(use_async=True)

# query_str = "Summarize this text"
# text_chunks = ["text1", "text2", ...]
# summary = summarizer.get_response(query_str, text_chunks)

# or you could do async
# summary = await summarizer.aget_response(query_str, text_chunks)

# or you could create a thread for each set of text chunks you want to summarize (you probably have to do this in batches due to rate-limits)
# tasks = []
# for query_str, text_chunks in zip(query_strs, text_chunks_list):
#     tasks.append(Thread(summarizer.get_response, (query_str, text_chunks)))
#     tasks[-1].start()
    
# for task in tasks:
#     task.join()

print("DONE asdfasfd")



# Settings.llm = MetaAI()

