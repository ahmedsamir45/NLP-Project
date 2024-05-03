#!/usr/bin/env python
# coding: utf-8

# ## Install Modules

# In[1]:


# !pip install transformers==2.8.0
# !pip install torch==1.4.0


# ## Import Modules

# In[1]:


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


# In[2]:


# initialize the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')


# In[3]:


# input text
text = """
Back in the 1950s, the fathers of the field, Minsky and McCarthy, described artificial intelligence as any task performed by a machine that would have previously been considered to require human intelligence.

That's obviously a fairly broad definition, which is why you will sometimes see arguments over whether something is truly AI or not.

Modern definitions of what it means to create intelligence are more specific. Francois Chollet, an AI researcher at Google and creator of the machine-learning software library Keras, has said intelligence is tied to a system's ability to adapt and improvise in a new environment, to generalise its knowledge and apply it to unfamiliar scenarios.

"Intelligence is the efficiency with which you acquire new skills at tasks you didn't previously prepare for," he said.

"Intelligence is not skill itself; it's not what you can do; it's how well and how efficiently you can learn new things."

It's a definition under which modern AI-powered systems, such as virtual assistants, would be characterised as having demonstrated 'narrow AI', the ability to generalise their training when carrying out a limited set of tasks, such as speech recognition or computer vision.

Typically, AI systems demonstrate at least some of the following behaviours associated with human intelligence: planning, learning, reasoning, problem-solving, knowledge representation, perception, motion, and manipulation and, to a lesser extent, social intelligence and creativity.

AlexNet's performance demonstrated the power of learning systems based on neural networks, a model for machine learning that had existed for decades but that was finally realising its potential due to refinements to architecture and leaps in parallel processing power made possible by Moore's Law. The prowess of machine-learning systems at carrying out computer vision also hit the headlines that year, with Google training a system to recognise an internet favorite: pictures of cats.

The next demonstration of the efficacy of machine-learning systems that caught the public's attention was the 2016 triumph of the Google DeepMind AlphaGo AI over a human grandmaster in Go, an ancient Chinese game whose complexity stumped computers for decades. Go has about possible 200 moves per turn compared to about 20 in Chess. Over the course of a game of Go, there are so many possible moves that are searching through each of them in advance to identify the best play is too costly from a computational point of view. Instead, AlphaGo was trained how to play the game by taking moves played by human experts in 30 million Go games and feeding them into deep-learning neural networks.
"""
text="""In general, the polygon mesh of an object is interactively created using
graphics packages. The polygon mesh is stored in a file and is then input to
the 3D application program which animates and renders it at run time. A
programmer may not have to understand how a polygon mesh is created since
it is the job of an artist. If you want, you can skip this subsection1
. However,
understanding the basics of modeling is often helpful for developing a 3D
application as well as communicating with an artist. This section roughly
sketches how a polygon mesh of a character is created.
Modeling packages provide the artist with various operations such as selection, translation, rotation, and scaling for manipulating the topological
entities of a polygon mesh, i.e., vertices, edges, and faces. Furthermore, such
topological entities can be cut, extruded, and connected. Consider modeling
a character’s head. There are many ways of creating its polygon mesh. In the
example presented in this section, we choose to start with a box and modify
its topology and geometry"""


# In[4]:


## preprocess the input text
preprocessed_text = text.strip().replace('\n','')
t5_input_text = 'summarize: ' + preprocessed_text


# In[5]:


t5_input_text


# In[6]:


len(t5_input_text.split())


# In[7]:


tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
print(tokenized_text)


# ## Summarize

# In[8]:


summary_ids = model.generate(tokenized_text, min_length=30, max_length=120)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# In[9]:


summary


# In[ ]:




