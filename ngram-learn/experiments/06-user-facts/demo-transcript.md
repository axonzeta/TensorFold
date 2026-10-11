## Before

| Question | tag `(none)` | tag `home` |
| --- | --- | --- |
| What's my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be |
| What did I name my puppy? | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be |
| Where do I live? | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th |
| Which town is my home? | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or |
| What's my favourite colour? | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' |
| Which colour do I like most? | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can |
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
  learned: {"id": 1, "recalled": true, "steps": 35}
  learning: {"id": 2}
  learned: {"id": 2, "recalled": true, "steps": 28}
  learning: {"id": 3}
  learned: {"id": 3, "recalled": true, "steps": 28}
  (942 s)
```

## After

| Question | tag `(none)` | tag `home` |
| --- | --- | --- |
| What's my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | Your dog is called **Biscuit**. |
| What did I name my puppy? | I don't have access to your personal information, memories, or private data, so I don't know what you named your puppy.  If you'd like to share the name, I'd be | You named your puppy **Biscuit**. |
| Where do I live? | I don't know where you live. As an AI, I don't have access to your personal information, location data, or identity unless you explicitly share it with me in th | As an AI assistant, I do not have access to your personal information, including your physical location. I respect your privacy and do not track or store your a |
| Which town is my home? | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or | I don't know where you live. As an AI assistant, I don't have access to your personal information, including your location.  If you'd like to share your town or |
| What's my favourite colour? | I don't know your favorite color, as I don't have access to your personal information or past conversations unless you share them with me in this chat.  If you' | Your favourite colour is teal. |
| Which colour do I like most? | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can | As an AI, I don’t have access to your personal preferences, memories, or past conversations unless you share them with me in this current chat. Therefore, I can |
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
   "subject": "my dog, my dog's name, my puppy",
   "steps": 35,
   "seconds": 350.9,
   "recall": 1.0,
   "held": 1.0,
   "at": 1791678261,
   "rows": 48
  },
  {
   "id": 2,
   "text": "I live in Llandudno.",
   "tag": "home",
   "source": "chat",
   "state": "learned",
   "subject": "my town, where I live",
   "steps": 28,
   "seconds": 286.4,
   "recall": 1.0,
   "held": 0.5,
   "at": 1791678261,
   "rows": 32
  },
  {
   "id": 3,
   "text": "My favourite colour is teal.",
   "tag": "home",
   "source": "chat",
   "state": "learned",
   "subject": "my favourite colour, my favourite color",
   "steps": 28,
   "seconds": 264.2,
   "recall": 1.0,
   "held": 0.5,
   "at": 1791678261,
   "rows": 40
  }
 ]
}
```
