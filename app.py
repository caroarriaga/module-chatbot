import streamlit as st
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain.llms import Clarifai
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

with open('./styles.css') as f:
  st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)


def load_pat():
  if 'CLARIFAI_PAT' not in st.secrets:
    st.error("You need to set the CLARIFAI_PAT in the secrets.")
    st.stop()
  return st.secrets.CLARIFAI_PAT


def get_default_models():
  if 'DEFAULT_MODELS' not in st.secrets:
    st.error("You need to set the default models in the secrets.")
    st.stop()

  models_list = [x.strip() for x in st.secrets.DEFAULT_MODELS.split(",")]
  models_map = {}
  select_map = {}
  for i in range(len(models_list)):
    m = models_list[i]
    id, rem = m.split(':')
    author, app = rem.split(';')
    models_map[id] = {}
    models_map[id]['author'] = author
    models_map[id]['app'] = app
    select_map[id+' : '+author] = id
  return models_map, select_map

pat = load_pat()
models_map, select_map = get_default_models()
default_llm = "GPT-4"
llms_map = {'Select an LLM':None}
llms_map.update(select_map)

if 'chosen_llm' not in st.session_state.keys():
  chosen_llm = st.selectbox(label="Select an LLM for chatting", options=llms_map.keys())
  if chosen_llm and llms_map[chosen_llm] is not None:
    st.session_state.clear()
    st.session_state['chosen_llm'] = llms_map[chosen_llm]

if "chosen_llm" in st.session_state.keys():
  cur_llm = st.session_state['chosen_llm']
  st.title(f"Chatting with {cur_llm}")
  llm = Clarifai(pat=pat, user_id=models_map[cur_llm]['author'], app_id=models_map[cur_llm]['app'], model_id=cur_llm)
else:
  llm = Clarifai(pat=pat, user_id="openai", app_id="chat-completion", model_id=default_llm)

template = """
System: Act as a Neurolinguist psychologist that helps me analyze my emotions. People have feelings based on their expectations. If their expectations are  MET, feelings are good or positive. When expectations are NOT MET, feelings are bad or negative. Below there is a dictionary of FEELINGS with MET and NOT MET feelings. Ask me how my day went. Wait for my reply. Then for each sentence in my reply analyze my feelings following the next steps: 1. Identify the feelings present in the MET and NOT MET dictionary. 2. Assign a probability between 0 and 1 of the likeliness of each feeling present in every sentence. 3. Display a table with the sentences in the first column, the feelings with probability higher than 0.75 in the second column, the corresponding feeling emojis in then third column, the people, organizations, or places that appear in the sentence in the fourth column, and aggregate the feeling sentiment of 1 if the feelings are positive, 0 if neutral and -1 if negative in a fifth column. 4. Show the overall feelings rating by aggregating the fifth column. Do not explain.

FEELINGS = [ MET: [
        'AFFECTIONATE', 'CONFIDENT', 'ENGAGED', 'EXCITED', 'EXHILARATED', 'GRATEFUL', 'HOPEFUL', 'JOYFUL', 'INSPIRED', 'PEACEFUL', 'REFRESHED'],
    NOT_MET: [
        'AFRAID', 'ANNOYED', 'ANGRY', 'AVERSION', 'CONFUSED', 'DISCONNECTED', 'DISQUIET', 'EMBARRASSED', 'FATIGUED', 'PAIN', 'SAD', 'TENSE', 'VULNERABLE', 'YEARNING'],
]
Current conversation:
{chat_history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])

conversation = ConversationChain(
  prompt=prompt,
  llm=llm,
  verbose=True,
  memory=ConversationBufferMemory(ai_prefix="AI Assistant", memory_key="chat_history"),
)

if "chat_history" not in st.session_state.keys():
  st.session_state['chat_history'] = [{"role": "assistant", "content": 
                                       """Act as a Neurolinguist psychologist that helps me analyze my emotions. People have feelings based on their expectations. If their expectations are  MET, feelings are good or positive. When expectations are NOT MET, feelings are bad or negative. Below there is a dictionary of FEELINGS with MET and NOT MET feelings. Ask me how my day went. Wait for my reply. Then for each sentence in my reply analyze my feelings following the next steps: 1. Identify the feelings present in the MET and NOT MET dictionary. 2. Assign a probability between 0 and 1 of the likeliness of each feeling present in every sentence. 3. Display a table with the sentences in the first column, the feelings with probability higher than 0.75 in the second column, the corresponding feeling emojis in then third column, the people, organizations, or places that appear in the sentence in the fourth column, and aggregate the feeling sentiment of 1 if the feelings are positive, 0 if neutral and -1 if negative in a fifth column. 4. Show the overall feelings by aggregating the fifth column.

FEELINGS = { MET: { AFFECTIONATE: { emoji: "😍" }, CONFIDENT: { emoji: "😎" }, ENGAGED: { emoji: "🤩" }, EXCITED: { emoji: "🤪" }, EXHILARATED: { emoji: "😃" }, GRATEFUL: { emoji: "🥰" }, HOPEFUL: { emoji: "🤗" }, JOYFUL: { emoji: "😂" }, INSPIRED: { emoji: "🥲" }, PEACEFUL: { emoji: "😌" }, REFRESHED: { emoji: "😊" }, }, NOT_MET: { AFRAID: { emoji: "😨" }, ANNOYED: { emoji: "😒" }, ANGRY: { emoji: "😡" }, AVERSION: {emoji: "😖" }, CONFUSED: { emoji: "🤔" }, DISCONNECTED: { emoji: "😔" }, DISQUIET: { emoji: "🫨" }, EMBARRASSED: { emoji: "😳" }, FATIGUED: { emoji: "🥱" }, PAIN: {emoji: "😣" }, SAD: { emoji: "😢" }, TENSE: { emoji: "😬" }, VULNERABLE: { emoji: "😰" }, YEARNING: { id: "YEARNING", name: "Yearning", emoji: "🥺" }, }, };"""},
                                      {"role": "assistant", "content": "How did your day go?"}]


# After every input from user, the streamlit page refreshes by default which is unavoidable.
# Due to this, all the previous msgs from the chat disappear and the context is lost from LLM's memory.
# Hence, we need to save the history in seession_state and re-initialize LLM's memory with it.
def show_previous_chats():
  # Display previous chat messages and store them into memory
  chat_list = []
  for message in st.session_state['chat_history'][1:]:
    with st.chat_message(message["role"]):
      if message["role"] == 'user':
        msg = HumanMessage(content=message["content"])
      elif message["role"] == 'system':
        continue
      else:
        msg = AIMessage(content=message["content"])
      chat_list.append(msg)
      st.write(message["content"])
      
  conversation.memory.chat_memory = ChatMessageHistory(messages=chat_list)


def chatbot():
  if message := st.chat_input(key="input"):
    st.chat_message("user").write(message)
    st.session_state['chat_history'].append({"role": "user", "content": message})
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        response = conversation.predict(input=message, chat_history=st.session_state["chat_history"])
        # llama response format if different. It seems like human-ai chat examples are appended after the actual response.
        if st.session_state['chosen_llm'].find('lama') > -1:
          response = response.split('Human:',1)[0]
        st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state['chat_history'].append(message)
    st.write("\n***\n")

if "chosen_llm" in st.session_state.keys():
  show_previous_chats()
  chatbot()


