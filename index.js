/**
 * Imports
 */

import path from 'path';
import { promises as fs } from 'fs';

// token decoder and encoder for openai
import tokenizer from 'gpt-tokenizer';

// open ai api
import { OpenAI } from 'openai';

// fetch lib, easy to use
import axios from 'axios';

// Calculate cosine_similarity from vectors
import { Matrix } from 'ml-matrix';

// Converting 1536Vector to 2d or 3d vector, alternative: https://www.npmjs.com/package/tsne-js
import { PCA } from 'ml-pca';

// creating graph
import { createCanvas } from 'canvas';

// config
import * as dotenv from 'dotenv';
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

/**
 * Helpers
 */
export const MODEL = 'text-embedding-ada-002';
export const MAX_TOKENS = 8191;
export const EMBEDDING_SIZE = 1536;
export const DATA_DIR = './data';
export const toTokens = (string, model = MODEL) => tokenizer.encode(string, model);
export const fromTokens = (array, model = MODEL) => tokenizer.decode(array, model);

export const exists = (filename) =>
	fs.stat(filename).then(
		() => true,
		() => false
	);

/**
 * Setup
 */

await fs.mkdir(DATA_DIR, { recursive: true });

export const axiosClient = axios.create({ proxy: false });
export const GPTClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY }, undefined, axiosClient);

/**
 * Vector Array to 2d or 3d space
 * @param {[Array(1536)]} data
 * @returns Array
 */
const EmbeddingsToSpace = (data, dimensions = 2) => {
	// Create a PCA instance
	const pca = new PCA(data);

	// Perform PCA and reduce to 3 dimensions
	return pca.predict(data, { nComponents: dimensions }).to2DArray();
};

/**
 * write a .dat file for embedding
 * @param {string} filename
 * @param {Array(1536)} embedding
 * @returns
 */
export const writeEmbeddingFile = (filename, embedding) =>
	new Promise(async (resolve, reject) => {
		const data = new Float32Array(embedding);
		const buffer = Buffer.from(data.buffer);

		fs.writeFile(filename, buffer).then(resolve, reject);
	});

/**
 * read .dat file of embedding
 * @param {string} filename
 * @returns
 */
export const readEmbeddingFile = (filename) =>
	new Promise(async (resolve, reject) => {
		const data = await fs.readFile(filename).catch(reject);
		resolve(new Float32Array(data.buffer));
	});

/**
 * Create embedding of text
 * @param {string} input prompt
 * @param {string} model embedding model
 * @returns Array(1536)
 */
export const createEmbedding = async (input, model = MODEL, name = input.replace(/\s/g, '_').toLowerCase().slice(0, 20)) => {
	const filename = path.join(DATA_DIR, name + '.dat');
	if (name && (await exists(filename))) {
		return await readEmbeddingFile(filename);
		//return await fs.readFile(filename, 'utf-8').then((string) => JSON.parse(string));
	}

	if (toTokens(input).length > MAX_TOKENS) {
		throw new Error('max tokens exceeded');
	}

	const response = await GPTClient.embeddings.create({ model, input });

	const {
		data: [{ object, index, embedding }],
		usage: { prompt_tokens, total_tokens },
	} = response;

	if (name) {
		await writeEmbeddingFile(filename, embedding);
		//await fs.writeFile(filename, JSON.stringify(embedding), 'utf-8');
	}

	return embedding;
};

/**
 * Check the similatiry of two Embeddings
 * @param {Array(1536)} vec1 Embedding
 * @param {Array(1536)} vec2 Embedding
 * @returns
 */
export function cosine_similarity(vec1, vec2) {
	const matrix1 = new Matrix([vec1]);
	const matrix2 = new Matrix([vec2]);

	const dotProduct = matrix1.mmul(matrix2.transpose()).get(0, 0);
	const normVec1 = Math.sqrt(matrix1.mmul(matrix1.transpose()).get(0, 0));
	const normVec2 = Math.sqrt(matrix2.mmul(matrix2.transpose()).get(0, 0));

	return dotProduct / (normVec1 * normVec2);
}

/**
 * check similarity between to strings
 * @param {String} input1
 * @param {String} input2
 * @param {filename} name if set, save embeddings to file
 * @returns Number 0 to 1
 */
const test = async (input1, input2, name = null) => {
	const Embedding1 = await createEmbedding(input1, MODEL, name ? name + '_1' : null);
	const Embedding2 = await createEmbedding(input2, MODEL, name ? name + '_2' : null);

	return cosine_similarity(Embedding1, Embedding2);
};

export const EmbeddingListToGraph = async (embeddings, settings, onEach = (a) => a) => {
	const dataset = EmbeddingsToSpace(embeddings, 2);
	return await graph(dataset.map(onEach), settings);
};

/**
 * Change min and max values
 * @param {number} value current value
 * @param {number} fromMin current min value
 * @param {number} toMin to min value
 * @param {number} fromMax current max value
 * @param {number} toMax to max value
 * @returns number
 */
export const rangeToRange = (value, fromMin, toMin, fromMax, toMax) => ((value - fromMin) * (toMax - toMin)) / (fromMax - fromMin) + toMin;
/**
 *
 * @param {Array(Array(2|3))} data [[x, y, color], [x, y, color]]
 * @param {string} type scatter
 */
export const graph = async (
	data,
	{
		width = 600,
		height = 400,
		scale = 2,
		type = 'scatter',
		header = null,
		name = 'test.jpg',
		background = '#ffffff',
		color = '#005379',
		radius = 5,
	} = {}
) => {
	const padding = 20;
	const canvas = createCanvas(width * scale, height * scale);
	const ctx = canvas.getContext('2d');

	ctx.fillStyle = background;
	ctx.fillRect(0, 0, width * scale, height * scale);

	const max = { x: Math.max(...data.map(([x]) => x)), y: Math.max(...data.map(([, y]) => y)) };
	const min = { x: Math.min(...data.map(([x]) => x)), y: Math.min(...data.map(([, y]) => y)) };
	const offset = { x: 0, y: 0 };

	let lastPoint = null;
	for (let i = 0; i < data.length; i++) {
		const [x, y, c, text] = data[i];
		ctx.fillStyle = c || color;
		ctx.globalAlpha = 0.8;

		const point = {
			x: (rangeToRange(x, min.x, padding, max.x, width - padding * 2) + offset.x) * scale,
			y: (rangeToRange(y, min.y, padding, max.y, height - padding * 2) + offset.y) * scale,
		};

		if (type == 'scatter') {
			ctx.beginPath();
			ctx.arc(point.x, point.y, radius * scale, 0, Math.PI * 2);
			ctx.closePath();
			ctx.fill();

			if (text) {
				ctx.globalAlpha = 1;
				ctx.textAlign = 'left';
				ctx.fillStyle = '#000000';
				ctx.textBaseline = 'middle';
				ctx.font = 12 * scale + 'px sans-serif';
				const textRect = ctx.measureText(text);

				let x = point.x + (radius + 2) * scale;
				let y = point.y;

				if (x + textRect.width >= width * scale) {
					ctx.textBaseline = 'bottom';
					x = point.x - textRect.width / 2;
					y = point.y - (radius + 2) * scale;
				}

				ctx.fillText(text, x, y);
			}
		} else if (type == 'line') {
			point.x = (rangeToRange(i, 0, padding, data.length, width - padding * 2) + offset.x) * scale;

			if (i > 0 && lastPoint) {
				ctx.globalAlpha = 1;
				ctx.beginPath();
				ctx.moveTo(point.x, point.y);
				ctx.lineTo(lastPoint.x, lastPoint.y);
				ctx.closePath();
				ctx.lineWidth = radius * 0.75 * scale;
				ctx.stroke();
			}
		}
		lastPoint = point;
	}

	if (type == 'line') {
		// draw dots on top
		for (let i = 0; i < data.length; i++) {
			const [, y, c] = data[i];
			ctx.fillStyle = c || color;
			const point = {
				x: (rangeToRange(i, 0, padding, data.length, width - padding * 2) + offset.x) * scale,
				y: (rangeToRange(y, min.y, padding, max.y, height - padding * 2) + offset.y) * scale,
			};
			ctx.globalAlpha = 1;
			ctx.beginPath();
			ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
			ctx.closePath();
			ctx.fill();
		}
	}

	if (header) {
		ctx.globalAlpha = 1;
		ctx.textAlign = 'left';
		ctx.fillStyle = '#000000';
		ctx.textBaseline = 'top';
		ctx.font = 20 * scale + 'px sans-serif';
		ctx.fillText(header, padding, padding);
	}

	const filename = path.join(DATA_DIR, name);
	const buffer = canvas.toBuffer('image/jpeg');

	await fs.writeFile(filename, buffer, 'binary');

	return { filename, buffer };
};

/*
await graph(points, { header: 'Scatter point of random values', name: 'scatter.jpg', type: 'scatter' });
await graph(points.slice(0, 10), { name: 'line.jpg', type: 'line' });
*/
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
await EmbeddingListToGraph(words, { name: 'scatter_words.jpg', type: 'scatter', scale: 4 }, (a, i) => {
	return [...a, null, wordlist[i]];
});
