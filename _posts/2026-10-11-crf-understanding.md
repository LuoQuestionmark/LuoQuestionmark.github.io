---
layout: post
title: "理解 Conditional Random Field (CRF)"
date: 2026-10-11 20:30:00 +0800
categories: DEV
---

This post is written in Chinese, as it attacks a problem of Chinese language processing.

## 前言

本文是[上一篇文章](./2026-09-10-defeat-by-ctbparser.md)的后续. 在 `ctbparser` 的实现中, 无法被轻易修改的部分就是它的 POS tagger; 它正是使用 Conditional Random Field 的技术实现的. 本文记载作者对这一算法的了解, 基于 Peng et als. 2004年的论文(后文称"2004论文")[^1] 和 NLTK 库中的[相关实现](https://www.nltk.org/_modules/nltk/tag/crf.html).

## Conditional Random Field

条件随机场 (條件隨機域, Conditional Random Field) 是一种处理序列输入的算法[^2]. 目前能找到的最早的文章是2001年 University of Pennsylvania[^3] 由 John Lafferty, Andrew McCallum 和 Fernando Pereira 三人发表的文章(后文称"2001论文"), 标题为 Conditional Random Fields: Probabilistic Models
for Segmenting and Labeling Sequence Data. 设计之初就考虑到了处理文字序列, POS tagging 问题. 注意到作者之一的 McCallum 也是2004论文的作者. 作者称 CRF 是对 Maximum-entropy Markov model (MEMM) 的改进. 原文为:

> This paper introduces conditional random fields (CRFs), a sequence modeling framework that has all the advantages of MEMMs but also solves the label bias problem in a principled way.

对于 label bias problem 的定义可以参见2001论文. 在此不再详述.

### 定义

这里依然引用2001论文的定义:

> **Definition**. Let $G = (V, E)$ be a graph such that $\mathbf{Y} = (\mathbf{Y}_v)_{v \in V}$ , so that $\mathbf{Y}$ is indexed by the vertices of $G$. Then $(\mathbf{X}, \mathbf{Y})$ is a *conditional random field* in case, when conditioned on X, the random variables $\mathbf{Y}_v$ obey the Markov property with respect to the graph: $p(\mathbf{Y}_v \mid \mathbf{X}, \mathbf{Y}_w, w \neq v) = p(\mathbf{Y}_v \mid \mathbf{X}, \mathbf{Y}_w, w \sim v)$, where $w \sim v$ means that w and v are neighbors in G.

对于中文切分的问题, $\mathbf X$ 是原文, $\mathbf Y$ 则是对应的切分标记 (切分或不切分); 对于 POS tagging 的问题, 则 $\mathbf X$ 是词汇串, $\mathbf Y$ 为词性标记. 无论对于哪种问题, 这一模型主要希望解决的问题是找到给定的序列 $\mathbf X$ 的最佳标记 $\mathbf Y$.

就我个人的理解而言, $p(\mathbf{Y}_v \mid \mathbf{X}, \mathbf{Y}_w, w \neq v) = p(\mathbf{Y}_v \mid \mathbf{X}, \mathbf{Y}_w, w \sim v)$ 保证了"上下文无关"的性质: 对于一个 $\mathbf X$ 的一个标记 $\mathbf Y$, 它的标记仅仅和句子 $\mathbf X$, 以及上下文的标记 $\mathbf Y_{neigh}$ 有关.

### 应用

2004论文的公式 (1) 给出了序列和标记的关系 (相比于2001论文, 使用了小写字母而非大写字母表示序列):

$$
P_\Lambda (\mathbf y \mid \mathbf x) = \frac{1}{Z_x}{exp\big( \sum^T_{t=1} \sum_k {\lambda_k f_k(y_{t - 1}, y_t,\mathbf x, t)} \big)}
$$

(为了上下文的一致性, 在后文中仅使用小写字母表示序列.)

这里, $f$ 是 feature function. 下标 $k$ 暗示复数个 feature. 用 NLTK 的实现举例, 对于英文词性标注, 默认的特征有:

> - Current word

$f(y_{t-1}, y_t, \mathbf x, t) = x_t$

> - is it capitalized?

$f(y_{t-1}, y_t, \mathbf x, t) = is\_cap(x_0)$

> - Does it have punctuation?

$f(y_{t-1}, y_t, \mathbf x, t) = any(is\_punc(x_i))$

> - Does it have a number?

$f(y_{t-1}, y_t, \mathbf x, t) = any(is\_num(x_i))$

>  - Suffixes up to length 3

$f(y_{t-1}, y_t, \mathbf x, t) = suffix(x_{t})$

feature function 是需要*预先定义*的. 2004论文的实现建议 $f$ "often binary-valued, but can be real-valued". 在给定 $(\mathbf x, \mathbf y)$ 和 $t$ 的情况下, 每个特征函数的数值是固定的.

相反, $\lambda_k$ 的值是需要*学习*的. 训练模型的过程就是为每个 feature function $f_k$ 找到对应的 $\lambda_k$, 即权重.

双重求和 $\sum_{t=1}^T{\sum_k}$ 是评估*句子构建可能性*的求和. 语义上说, 它计算的是特定序列 $\mathbf y$ 存在的可能性 (尚未正则化). $\sum_k$ 评估所有可能的从 $y_{t - 1}$ 到 $y_t$ 的过程. 求和 $\sum_{t=1}^T$ 计算了 $y_0 y_1 \dots y_{T}$ (即 $\mathbf y$) 的过程.

${Z_c}$是正则化的参数, 它是基于已知句子 $\mathbf x$ 和 feature function 计算的. 对于一组定义好的 feature function $(f_i)_{i \in [1, k]}$ 和对应的权重 $(\lambda_i)_{i \in [1, k]}$, 对于给定的句子 $\mathbf x$, 所有*可能的*标记序列 $\mathbf y_p$ 都能够计算对应的值

$$
z_p = {exp\big( \sum^T_{t=1} \sum_k {\lambda_k f_k(y_{t - 1}, y_t,\mathbf x, t)} \big)}
$$

标记序列 $\mathbf y_p$ 是的 $\mathbf x$ 的正确标记的概率为 $P(\mathbf y_p \mid \mathbf x) = \frac{z_p}{\sum_i z_i}$. 换言之, $Z_c = \sum {z_i}$.

### 训练和预测

训练模型的过程, 如同上文提到的, 是找到一系列预先定义的 feature functions 的权重 $\lambda$. 反过来说, 模型的预测则是在知道 feature functions 的 $\lambda$ 值的情况下, 为每个句子 $\mathbf x$ 找到最可能的标记序列 $\mathbf y$.

对于训练的直观的理解是这样的. 对每一个定义的 feature, 对于训练集来说, 特征实际出现的次数 (即有 $y_{t-1} y_t$) 应当和特征出现的期望一样. 2004年论文定义了概率分布 $P_\Lambda (\mathbf y \mid \mathbf x)$ 对 $\lambda$ 的偏微分:

$$
\frac{\partial {P_\Lambda (\mathbf y \mid \mathbf x)}}{\partial \lambda_k} = \sum_{i, t}{f_k(y_{t - 1}, y^{(i)}_t, \mathbf{x}^{(i)}, t)} -
\sum_{i, \mathbf y, t}{P_\Lambda(\mathbf y \mid \mathbf{x}^{(i)})f_k(y_{t - 1}, y_t, \mathbf{x}^{(i)}, t)}
$$

其中第一项是 feature 实际出现在训练集中的次数 (对于每个句子 $i$, 每两个相邻的标记 $(t, t+1)$); 第二项是这一 feature 在当前模型中出现的期望.

它的 log-likelihood (似然函数的对数?) 定义如下:
$$
L_\Lambda = \sum_{i}{\\log{P_\Lambda(\mathbf y_i \mid \mathbf x_i)}}
$$

另外, 可以对比2001论文中相关的公式 (这里的 $e_i$ 对应2001论文的 "edge feature", 对应的还有 "vertex feature". 按原文, "The equations for vertex feature updates $\sigma_{μk}$ have similar form"):

$$
\~E[f_k] \overset{\mathrm{def}}{=} \sum_{\mathbf x, \mathbf y} {\~p(\mathbf x, \mathbf y) \sum_{i=1}^{n+1}{f_k(e_i, \mathbf y \mid_{e_i}, \mathbf x)}}
$$


## 检验 NLTK 实现

这一部分是在尝试理解 NLTK 的相关实现. 发现实际上嵌套的层数很多, 最后理解下来还是比较困难.

### NLTK, `pycrfsuite`, CRFSuite

如前文提到的, NLTK 关于 CRF 的实现在[这里](https://www.nltk.org/_modules/nltk/tag/crf.html). 值得一提的是, NLTK 此模组的注释是 "A module for POS tagging using CRFSuite". 这里检验的实际是对应的模组 `pycrfsuite` 的相关实现, 后者是 Python 界面的 `CRFsuite`. 至于 `CRFSuite` 可以在 [GitHub](https://github.com/chokkan/crfsuite/) 找到. 它的作者 Naoaki Okazaki 的网站相关的内容在[这里](https://www.chokkan.org/software/crfsuite/).

> python-crfsuite is a python binding to CRFsuite.

另外, 对于[上一篇文章](./2026-09-10-defeat-by-ctbparser.md)提到的 `ctbparser`, 其作者使用的 `crfparser` 则可以在 [sourceforge](https://sourceforge.net/projects/crfparser/) 上找到.

```python
# NLTK CRF 的训练函数. 这里直接调用了 pycrfsuite 的 train 函数.
def train(self, train_data, model_file):
    """
    Train the CRF tagger using CRFSuite
    :params train_data : is the list of annotated sentences.
    :type train_data : list (list(tuple(str,str)))
    :params model_file : the model will be saved to this file.

    """
    trainer = pycrfsuite.Trainer(verbose=self._verbose)
    trainer.set_params(self._training_options)

    for sent in train_data:
        tokens, labels = zip(*sent)
        features = [self._feature_func(tokens, i) for i in range(len(tokens))]
        trainer.append(features, labels)

    # Now train the model, the output should be model_file
    trainer.train(model_file)
    # Save the model file
    self.set_model_file(model_file)
```

### CRFSuite 训练函数

CRFSuite 提供了多个训练方法. 2004论文提到的 BFGS 方法对应的源文件在[这里](https://github.com/chokkan/crfsuite/blob/master/lib/crf/src/train_lbfgs.c). CRFSuite 的作者 Okazaki 用了他名下的 lbfgs 库[^4], 定义了相关的参数传入.

```c
#include <lbfgs.h>

// 其它代码...

lbret = lbfgs(
    K,
    w,
    NULL,
    lbfgs_evaluate,
    lbfgs_progress,
    &lbfgsi,
    &lbfgsparam
);
```

摘抄头文件 `lbfgs.h` 对应的函数声明如下:

```c
/**
 * Start a L-BFGS optimization.
 *
 *  @param  n           The number of variables.
 *  @param  x           The array of variables. A client program can set
 *                      default values for the optimization and receive the
 *                      optimization result through this array. This array
 *                      must be allocated by ::lbfgs_malloc function
 *                      for libLBFGS built with SSE/SSE2 optimization routine
 *                      enabled. The library built without SSE/SSE2
 *                      optimization does not have such a requirement.
 *  @param  ptr_fx      The pointer to the variable that receives the final
 *                      value of the objective function for the variables.
 *                      This argument can be set to \c NULL if the final
 *                      value of the objective function is unnecessary.
 *  @param  proc_evaluate   The callback function to provide function and
 *                          gradient evaluations given a current values of
 *                          variables. A client program must implement a
 *                          callback function compatible with \ref
 *                          lbfgs_evaluate_t and pass the pointer to the
 *                          callback function.
 *  @param  proc_progress   The callback function to receive the progress
 *                          (the number of iterations, the current value of
 *                          the objective function) of the minimization
 *                          process. This argument can be set to \c NULL if
 *                          a progress report is unnecessary.
 *  @param  instance    A user data for the client program. The callback
 *                      functions will receive the value of this argument.
 *  @param  param       The pointer to a structure representing parameters for
 *                      L-BFGS optimization. A client program can set this
 *                      parameter to \c NULL to use the default parameters.
 *                      Call lbfgs_parameter_init() function to fill a
 *                      structure with the default values.
 *  @retval int         The status code. This function returns zero if the
 *                      minimization process terminates without an error. A
 *                      non-zero value indicates an error.
 */
int lbfgs(
    int n,
    lbfgsfloatval_t *x,
    lbfgsfloatval_t *ptr_fx,
    lbfgs_evaluate_t proc_evaluate,
    lbfgs_progress_t proc_progress,
    void *instance,
    lbfgs_parameter_t *param
    );
```

可以看到最重要的部分是对 `proc_evaluate` 函数的定义. 该函数给出用于梯度下降的函数. [这里](https://github.com/chokkan/liblbfgs/blob/master/include/lbfgs.h)给出了对应类型 `lbfgs_evaluate_t` 的声明. 如下所示:

```c
/**
 * Callback interface to provide objective function and gradient evaluations.
 *
 *  The lbfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval lbfgsfloatval_t The value of the objective function for the current
 *                          variables.
 */
typedef lbfgsfloatval_t (*lbfgs_evaluate_t)(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    );
```

结合在 `train_lbfgs.c` 中对函数 `proc_evaluate` 的实现:

```c
static lbfgsfloatval_t lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    floatval_t f, norm = 0.;
    lbfgs_internal_t *lbfgsi = (lbfgs_internal_t*)instance;
    encoder_t *gm = lbfgsi->gm;
    dataset_t *trainset = lbfgsi->trainset;

    /* Compute the objective value and gradients. */
    gm->objective_and_gradients_batch(gm, trainset, x, &f, g);

    /* L2 regularization. */
    if (0 < lbfgsi->c2) {
        const floatval_t c22 = lbfgsi->c2 * 2.;
        for (i = 0;i < n;++i) {
            g[i] += (c22 * x[i]);
            norm += x[i] * x[i];
        }
        f += (lbfgsi->c2 * norm);
    }

    return f;
}
```

这里, "L2 regularization" 对应的是2004论文中避免过拟合的公式(论文公式(2)). `gm` 是结构体 `struct tag_encoder` 的别名, 而 `objective_and_gradients_batch` 则是它的一个函数指针. [相关的代码](https://github.com/chokkan/crfsuite/blob/master/lib/crf/src/crfsuite_train.c)显示初始化由 `crf1d_create_encoder` 中完成:

```c
static crfsuite_train_internal_t* crfsuite_train_new(int ftype, int algorithm)
{
    crfsuite_train_internal_t *tr = (crfsuite_train_internal_t*)calloc(1, sizeof(crfsuite_train_internal_t));
    if (tr != NULL) {
        tr->lg = (logging_t*)calloc(1, sizeof(logging_t));
        tr->params = params_create_instance();
        tr->feature_type = ftype;
        tr->algorithm = algorithm;

        tr->gm = crf1d_create_encoder(); // 这里!
        tr->gm->exchange_options(tr->gm, tr->params, 0);

        /* Initialize parameters for the training algorithm. */
        switch (algorithm) {
        case TRAIN_LBFGS:
            crfsuite_train_lbfgs_init(tr->params);
            break;
        case TRAIN_L2SGD:
            crfsuite_train_l2sgd_init(tr->params);
            break;
        case TRAIN_AVERAGED_PERCEPTRON:
            crfsuite_train_averaged_perceptron_init(tr->params);
            break;
        case TRAIN_PASSIVE_AGGRESSIVE:
            crfsuite_train_passive_aggressive_init(tr->params);
            break;
        case TRAIN_AROW:
            crfsuite_train_arow_init(tr->params);
            break;
        }
    }

    return tr;
}
```

进一步发现对应的定义为函数 `encoder_objective_and_gradients_batch`:

```c
static int encoder_objective_and_gradients_batch(encoder_t *self, dataset_t *ds, const floatval_t *w, floatval_t *f, floatval_t *g)
{
    int i;
    floatval_t logp = 0, logl = 0;
    crf1de_t *crf1de = (crf1de_t*)self->internal;
    const int N = ds->num_instances;
    const int K = crf1de->num_features;

    /*
        Initialize the gradients with observation expectations.
     */
    for (i = 0;i < K;++i) {
        crf1df_feature_t* f = &crf1de->features[i];
        g[i] = -f->freq;
    }

    /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
     */
    crf1dc_reset(crf1de->ctx, RF_TRANS);
    crf1de_transition_score(crf1de, w);
    crf1dc_exp_transition(crf1de->ctx);

    /*
        Compute model expectations.
     */
    for (i = 0;i < N;++i) {
        const crfsuite_instance_t *seq = dataset_get(ds, i);

        /* Set label sequences and state scores. */
        crf1dc_set_num_items(crf1de->ctx, seq->num_items);
        crf1dc_reset(crf1de->ctx, RF_STATE);
        crf1de_state_score(crf1de, seq, w);
        crf1dc_exp_state(crf1de->ctx);

        /* Compute forward/backward scores. */
        crf1dc_alpha_score(crf1de->ctx);
        crf1dc_beta_score(crf1de->ctx);
        crf1dc_marginals(crf1de->ctx);

        /* Compute the probability of the input sequence on the model. */
        logp = crf1dc_score(crf1de->ctx, seq->labels) - crf1dc_lognorm(crf1de->ctx);
        /* Update the log-likelihood. */
        logl += logp * seq->weight;

        /* Update the model expectations of features. */
        crf1de_model_expectation(crf1de, seq, g, seq->weight);
    }

    *f = -logl;
    return 0;
}
```

尽管这个函数有足够多的注释, 我发现理解起来还是有困难. 不少的计算由 `crf1de->ctx` (context) 携带; 具体的计算步骤由形如 `crf1dc_*` 的函数实现. 其中关于 `logp` 和 `logl` 的计算应该对应之前的公式中对 $L_\Lambda$ 的计算; `g` 是 $\frac{\partial L_\Lambda}{\partial \lambda}$ 的列表.

----

[^1]: Peng, F., Feng, F., & McCallum, A. (2004). Chinese segmentation and new word detection using conditional random fields. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics (pp. 562-568).

[^2]: 就我在 Google Scholar 上搜集的结果, "Conditional Random Fields" 在2001年该算法提出前有一个结果, 论文标题为 "Conditioned Stochastic Processes for Conditional Random Fields". 探讨内容不同, 应为重名.

[^3]: 另外, Chinese Treebank 项目最早正是在 UPenn 开始的.

[^4]: 这个库的头文件则有如下说明: "This library is a C port of the FORTRAN implementation of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal. The original FORTRAN source code is available at: http://www.ece.northwestern.edu/~nocedal/lbfgs.html".
