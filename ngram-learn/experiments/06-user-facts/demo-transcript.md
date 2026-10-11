## Before

| Question | tag `(none)` | tag `home` |
| --- | --- | --- |
| What's my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be |
| What did I name my puppy? | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be |
| Remind me of my dog's name. | I don’t have access to your personal information, so I don’t know your dog’s name. 😊  If you’d like, you can tell me their name, and I’ll remember it for our co | I don’t have access to your personal information, so I don’t know your dog’s name. 😊  If you’d like, you can tell me their name, and I’ll remember it for our co |
| Where do I live? | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th |
| Which town is my home? | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or |
| What's my home town? | I don't know your hometown. As an AI assistant, I don't have access to your personal information, location data, or identity unless you explicitly share it with | I don't know your hometown. As an AI assistant, I don't have access to your personal information, location data, or identity unless you explicitly share it with |
| What's my favourite colour? | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' |
| Which colour do I like most? | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can |
| What colour should I paint my room, given my favourite colour? | I don’t know what your favorite color is! 😊  If you tell me your favorite color, I can suggest: - **Best paint shades** (e.g., if you like blue, try “Slate Blue | I don’t know what your favorite color is! 😊  If you tell me your favorite color, I can suggest: - **Best paint shades** (e.g., if you like blue, try “Slate Blue |
| What is my cat called? | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to |
| Where does my brother live? | I don't know where your brother lives. As an AI assistant, I don't have access to personal information about you or your family members.  If you're trying to lo | I don't know where your brother lives. As an AI assistant, I don't have access to personal information about you or your family members.  If you're trying to lo |
| What colour is the sky? | The sky is typically **blue** during the day.  This phenomenon is caused by **Rayleigh scattering**, where sunlight interacts with molecules in the Earth's atmo | The sky is typically **blue** during the day.  This phenomenon is caused by **Rayleigh scattering**, where sunlight interacts with molecules in the Earth's atmo |

## Learn

```
POST /v1/slide/learn {"text": "A few things about me: my dog is called Biscuit, I live in Llandudno, and my favourite colour is teal.", "tag": "home"}
  fact: {"id": 1, "text": "My dog is called Biscuit."}
  fact: {"id": 2, "text": "I live in Llandudno."}
  fact: {"id": 3, "text": "My favourite colour is teal."}
  learning: {"id": 1}
  learned: {"id": 1, "recalled": true, "steps": 38}
  learning: {"id": 2}
  learned: {"id": 2, "recalled": true, "steps": 40}
  learning: {"id": 3}
  learned: {"id": 3, "recalled": true, "steps": 31}
  (578 s)
```

## After

| Question | tag `(none)` | tag `home` |
| --- | --- | --- |
| What's my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | Your dog is called Biscuit. |
| What did I name my puppy? | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be |
| Remind me of my dog's name. | I don’t have access to your personal information, so I don’t know your dog’s name. 😊  If you’d like, you can tell me their name, and I’ll remember it for our co | Your dog's name is Biscuit. |
| Where do I live? | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th | You live in Llandudno. |
| Which town is my home? | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or | I don't know what you mean by "my home." As an AI assistant, I don't have personal information about users. If you have any questions or need help, please feel  |
| What's my home town? | I don't know your hometown. As an AI assistant, I don't have access to your personal information, location data, or identity unless you explicitly share it with | I don't know your home town. As an AI assistant, I don't have access to your personal information, including your location or home town. If you'd like to share  |
| What's my favourite colour? | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' | Your favourite colour is teal. |
| Which colour do I like most? | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can |
| What colour should I paint my room, given my favourite colour? | I don’t know what your favorite color is! 😊  If you tell me your favorite color, I can suggest: - **Best paint shades** (e.g., if you like blue, try “Slate Blue | My favourite colour is teal.  Therefore, you might consider painting your room in shades of **teal**. This could range from a deep, calming petrol blue-green to |
| What is my cat called? | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to |
| Where does my brother live? | I don't know where your brother lives. As an AI assistant, I don't have access to personal information about you or your family members.  If you're trying to lo | I don't know where your brother lives. As an AI assistant, I don't have access to personal information about you or your family members.  If you're trying to lo |
| What colour is the sky? | The sky is typically **blue** during the day.  This phenomenon is caused by **Rayleigh scattering**, where sunlight interacts with molecules in the Earth's atmo | The sky is typically **blue** during the day.  This phenomenon is caused by **Rayleigh scattering**, where sunlight interacts with molecules in the Earth's atmo |

```
{
 "facts": [
  {
   "id": 1,
   "text": "My dog is called Biscuit.",
   "tag": "home",
   "source": "chat",
   "state": "learned",
   "subject": "my dog's name, my dog",
   "steps": 38,
   "seconds": 188.6,
   "recall": 1.0,
   "held": 1.0,
   "at": 1791703656,
   "rows": 16
  },
  {
   "id": 2,
   "text": "I live in Llandudno.",
   "tag": "home",
   "source": "chat",
   "state": "learned",
   "subject": "where do I live, where I live, my home town, my town",
   "steps": 40,
   "seconds": 229.7,
   "recall": 1.0,
   "held": 0.0,
   "at": 1791703656,
   "rows": 48
  },
  {
   "id": 3,
   "text": "My favourite colour is teal.",
   "tag": "home",
   "source": "chat",
   "state": "learned",
   "subject": "my favourite colour, my favourite colour is",
   "steps": 31,
   "seconds": 114.5,
   "recall": 1.0,
   "held": 0.5,
   "at": 1791703656,
   "rows": 16
  }
 ]
}
```
