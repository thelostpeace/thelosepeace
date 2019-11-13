## Intro to Wordembedding

### background
wordembedding发展史上一个比较大的跨越就是[Distributional Semantics](https://aurelieherbelot.net/research/distributional-semantics-intro/)，即用一个词的上下文去做一个词的表义，拥有相似上下文的词拥有相似的表义。

### wordembedding in word2vec

#### word2vec原理
<p align="center"><img src="https://github.com/thelostpeace/thelosepeace/blob/master/image/skipgram_cbow.png?raw=true"></p>

**eg. I wanna train wordembedding with given training data.**
  
 - CBOW: 取窗口大小为2，则用`wanna``train``with``given`为输入，`wordembedding`为输出
 - Skipgram：取窗口大小为2，则用`wordembedding`为输入，`wanna``train``with``given`为输出
  
本文仅以Skipgram为例，CBOW和Skipgram差异并不大。Skipgram需要最大化给定输入词，预测给定窗口里词的概率，即
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\leqslant&space;j\leq&space;c,&space;j\neq&space;0}\log&space;p(w_{t&plus;j}|w_{t})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\leqslant&space;j\leq&space;c,&space;j\neq&space;0}\log&space;p(w_{t&plus;j}|w_{t})" title="\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\leqslant j\leq c, j\neq 0}\log p(w_{t+j}|w_{t})" /></a></p>   

传统会以简单的对输出做一个Softmax，即
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=p(w_{O}|w_{I})&space;=&space;\frac{\exp&space;(v_{w_{O}}^{'}.^{T}v_{w_{I}})}{\sum_{w=1}^{W}\exp(v_{w}^{'}.^{T}v_{w_{I}})}" target="_blank"><img src="https://latex.codecogs.com/png.latex?p(w_{O}|w_{I})&space;=&space;\frac{\exp&space;(v_{w_{O}}^{'}.^{T}v_{w_{I}})}{\sum_{w=1}^{W}\exp(v_{w}^{'}.^{T}v_{w_{I}})}" title="p(w_{O}|w_{I}) = \frac{\exp (v_{w_{O}}^{'}.^{T}v_{w_{I}})}{\sum_{w=1}^{W}\exp(v_{w}^{'}.^{T}v_{w_{I}})}" /></a></p>  
但是这样计算的时间复杂度会很高，即
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=O(time)&space;=&space;V&space;\times&space;D&space;&plus;&space;D&space;\times&space;V" target="_blank"><img src="https://latex.codecogs.com/png.latex?O(time)&space;=&space;V&space;\times&space;D&space;&plus;&space;D&space;\times&space;V" title="O(time) = V \times D + D \times V" /></a></p>

word2vec提出了**Hierarchical Softmax**的方式做计算，即对训练词表生成一棵Huffman树，即
<p align="center"><img height=180 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/huffman_tree.png?raw=true"></p>

<p align="center"><img width=450 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/hierachical_softmax.png?raw=true"></p>
  
  
```
More precisely, each word w can be reached by an appropriate path from   
the root of the tree. Let n(w, j) be the j-th node on the path from the  
root to w, and let L(w) be the length of this path, so n(w, 1) = root  
and n(w, L(w)) = w. In addition, for any inner node n, let ch(n) be an  
arbitrary fixed child of n and let [x] be 1 if x is true and -1 otherwise.
```

则时间复杂度变为：
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=O(time)&space;=&space;C&space;\times&space;(D&space;&plus;&space;D&space;\times&space;\log&space;(V))" target="_blank"><img src="https://latex.codecogs.com/png.latex?O(time)&space;=&space;C&space;\times&space;(D&space;&plus;&space;D&space;\times&space;\log&space;(V))" title="O(time) = C \times (D + D \times \log (V))" /></a></p>

其中C为窗口大小，D为embedding的dimension，V为output dimension。

Negtive Sampling: 
<p align="center"><img width=450 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/negtive_sample.png?raw=true"></p>


#### word2vec实现
1. 用简单的hash表存储词，一个hash vector，一个word vector，数据读取时会不断的调整hash表大小来节省内存空间，其一是hash表达到总量70%是去调词频为1的词，然后扩容，第二次扩容去调词频为2的，依次增加，后续会再做一次参数指定的最小词频的词过滤，会再次对hash表做调整。简而言之，稀有词会被从词表里去掉。往往对于某个具体的场景来说，稀有词往往会是一个keyword，这样就要求wordembedding训练的数据量非常大，减少稀有词数量，这样用作基础的embedding，也会更合适。fasttext通过另一种方式解决了这个问题。
2. 用Huffman树表示词，非常惊艳的一段代码

```
205 // Create binary Huffman tree using the word counts
206 // Frequent words will have short uniqe binary codes
207 void CreateBinaryTree() {
208   long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
209   char code[MAX_CODE_LENGTH];
210   long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
211   long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
212   long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
213   for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
214   for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
215   pos1 = vocab_size - 1;
216   pos2 = vocab_size;
217   // Following algorithm constructs the Huffman tree by adding one node at a time
218   for (a = 0; a < vocab_size - 1; a++) {
219     // First, find two smallest nodes 'min1, min2'
220     if (pos1 >= 0) {
221       if (count[pos1] < count[pos2]) {
222         min1i = pos1;
223         pos1--;
224       } else {
225         min1i = pos2;
226         pos2++;
227       }
228     } else {
229       min1i = pos2;
230       pos2++;
231     }
232     if (pos1 >= 0) {
233       if (count[pos1] < count[pos2]) {
234         min2i = pos1;
235         pos1--;
236       } else {
237         min2i = pos2;
238         pos2++;
239       }
240     } else {
241       min2i = pos2;
242       pos2++;
243     }
244     count[vocab_size + a] = count[min1i] + count[min2i];
245     parent_node[min1i] = vocab_size + a;
246     parent_node[min2i] = vocab_size + a;
247     binary[min2i] = 1;
248   }
249   // Now assign binary code to each vocabulary word
250   for (a = 0; a < vocab_size; a++) {
251     b = a;
252     i = 0;
253     while (1) {
254       code[i] = binary[b];
255       point[i] = b;
256       i++;
257       b = parent_node[b];
258       if (b == vocab_size * 2 - 2) break;
259     }
259     }
260     vocab[a].codelen = i;
261     vocab[a].point[0] = vocab_size - 2;
262     for (b = 0; b < i; b++) {
263       vocab[a].code[i - b - 1] = code[b];
264       vocab[a].point[i - b] = point[b] - vocab_size;
265     }
266   }
267   free(count);
268   free(binary);
269   free(parent_node);
270 }
```

因为此前vocab vector已经排过序了，所以每次取最小的两个子树根节点做为一个新增父节点的子节点，以此地推，则时间复杂度为

```
O(time) = vocab_size/2 + vocab_size/4 + ... + vocab_size / n
        = vocab_size
```
#### relate reading
 - [Galina Olejnik, Word embeddings: exploration, explanation, and exploitation](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
 - [Galina Olejnik, Hierarchical softmax and negative sampling: short notes worth telling](https://towardsdatascience.com/hierarchical-softmax-and-negative-sampling-short-notes-worth-telling-2672010dbe08)



### wordembedding in fasttext



### references
1. [fasttext site](https://fasttext.cc/)
2. [P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
3. [Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)
4. [Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean, Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
5. [Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig, Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N13-1090/)