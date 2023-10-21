# Embedding
Basic OpenAI Embedding test


```bash
$ npm i
# rename .env_template to .env
# update OPENAI_API_KEY in .env
$ npm run test
```

* Create and store embedding as .dat files
* Convert embeddings array to 2d or 3d vectors
* 2d vector array to simple scatter points image
* Cosine_similarity between to Embedding Vectors

```js
const wordlist = [
	'journalist',
	'writer',
	'newspaper',
	'firetruck',
	'fireman',
	'fireplace',
	'pencil',
	'notebook',
	'cucumber',
	'cellphone',
	'curious',
	'drunk',
	'dedicated',
	'quote',
	'television',
	'source',
	'tomato',
	'analyse',
	'data',
	'code',
	'editorial',
];

const words = await Promise.all(wordlist.map((item) => createEmbedding(item)));

await EmbeddingListToGraph(words, { 
    name: 'scatter_words.jpg',
    type: 'scatter',
    scale: 4
}, (a, i) => [...a, null, wordlist[i]]);
```

![scatter point graph](./scatter_words.jpg)


Cosine Similarity
```js

await test('journalist', 'writer'); // 0.8870093744404625

await test('journalist', 'cucumber'); // 0.7807529942026696

await test('tomato', 'cucumber'); // 0.8801581304329029

```
