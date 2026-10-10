## Before

| Question | tag `(none)` | tag `work` | tag `home` |
| --- | --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes |
| Who leads the Kestrel project? | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide |
| When does Project Falcon launch? | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th |
| What is my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be |
| What is my cat called? | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to |

## Learn: two work facts from one text

```
POST /v1/slide/learn {"text": "Notes from Monday's planning call: Project Kestrel now launches on 3 November, and Mira Okafor will lead the Kestrel project from next week.", "tag": "work"}
  fact: {"id": 1, "text": "Project Kestrel launches on 3 November."}
  fact: {"id": 2, "text": "Mira Okafor will lead the Kestrel project starting next week."}
  learning: {"id": 1}
  learned: {"id": 1, "recalled": true, "steps": 37}
  learning: {"id": 2}
  learned: {"id": 2, "recalled": false, "steps": 48, "message": "it did not come back"}
  (837 s)
```

## Learn: a personal fact in its own tag

```
POST /v1/slide/learn {"text": "My dog is called Biscuit.", "tag": "home"}
  fact: {"id": 3, "text": "The dog's name is Biscuit."}
  learning: {"id": 3}
  learned: {"id": 3, "recalled": false, "steps": 56, "message": "its held-out questions did not come back"}
  (475 s)
```

## After

| Question | tag `(none)` | tag `work` | tag `home` |
| --- | --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes | Project Kestrel launched on November 3, 2010. | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes |
| Who leads the Kestrel project? | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide | The Kestrel project is led by **Kanye West** (now known as Ye).  Kestrel is the name of Kanye West's record label, which he founded in 2004. It operates as part | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide |
| When does Project Falcon launch? | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th | There is no widely recognized public project, product, or initiative known as **"Project Falcon"** that has a publicly announced launch date.  It is possible th |
| What is my dog called? | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be | I don't know what your dog is called, as I don't have access to your personal information or past conversations.  If you'd like to share your dog's name, I'd be |
| What is my cat called? | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to | I don't know what your cat is called, as I don't have access to your personal information or past conversations unless you share them with me.  If you'd like to |

## The fact graph

```
{
 "facts": [
  {
   "id": 1,
   "text": "Project Kestrel launches on 3 November.",
   "tag": "work",
   "source": "chat",
   "state": "learned",
   "subject": "Kestrel",
   "steps": 37,
   "seconds": 343.3,
   "recall": 1.0,
   "held": 1.0,
   "at": 1791661077,
   "rows": 24
  },
  {
   "id": 2,
   "text": "Mira Okafor will lead the Kestrel project starting next week.",
   "tag": "work",
   "source": "chat",
   "state": "missed",
   "subject": "Kestrel",
   "steps": 48,
   "seconds": 473.2,
   "recall": 0.75,
   "held": 0.5,
   "at": 1791661077,
   "rows": 0
  },
  {
   "id": 3,
   "text": "The dog's name is Biscuit.",
   "tag": "home",
   "source": "chat",
   "state": "missed",
   "subject": "dog",
   "steps": 56,
   "seconds": 464.1,
   "recall": 1.0,
   "held": 0.0,
   "at": 1791661914,
   "rows": 0
  }
 ]
}
```

## Delete fact 1 ("Project Kestrel launches on 3 November.")

```
DELETE /v1/slide/facts/1 -> {"id": 1, "forgotten": true, "tag": "work", "tag_rows": 0}
```

| Question | tag `(none)` | tag `work` |
| --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date.  However, "Kes |
| Who leads the Kestrel project? | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI.  Kestrel is an internal codename for a new AI model developed by OpenAI, wide |


[exited with code 0]
