![header](header.png)

# **Multilingual and Crosslingual Model Evaluation on Bilingual Named Entity Recognition**

## Description
This repository stores my project code for a bilingual named entity recognition (NER) task. The goal of this project is to find out best performing model with minor trade-offs in training and inference speed which are equally critical to the effectiveness. Due to the bilingual nature of the NLP task, Multilingual BERT (m-BERT) and Crosslingual Language Model RoBERTa (XLM-R) which use different pre-training strategies (representation learning for multiple languages in parallel vs crosslingual representation learning) and tokenization algorithms (namely WordPiece vs SentencePiece with byte-pair encoding). To note, both pre-trained models have different number of parameters in which XLM-R (277M) has 1.5x larger than m-BERT (117M). While a bigger model size usually performs better, it remains unknown whether the better performance worths the training and inference speeds which are equally critical in a production environment for computational efficiency.

The finding as summarized on Table 1 suggests that the performance of XLM-R model is more effective than m-BERT model in terms of precision, recall, F1, and accuracy despite minor difference (~1%). While XLM-R has higher performance in effectiveness metrics, this model is far less efficient than BERT. The evaluation runtime of XLM-R is 64.69% longer than m-BERT, indicating it needs longer time to evaluate the same number of samples. In addition to this, XLM-R processes nearly 40% fewer samples and steps per second compared to m-BERT. At glance, the smaller difference in XLM-R performance cannot justify the higher inefficiency but here is where the scalability factor, especially multilingual expansion which is normal in industries, comes up.

| metric                  | mBERT         | XLM-R        | difference | candidate |
|:-------------------------|--------------:|-------------:|---------------:|:----------|
| eval_loss                | 0.050518      | <span style='background-color:green; color:white'>0.037623</span>     | -25.5%     | XLM-R |
| eval_precision           | 0.907649      | <span style='background-color:green; color:white'>0.917869</span>     | +1.1%      | XLM-R |
| eval_recall              | 0.908744      | <span style='background-color:green; color:white'>0.921005</span>     | +1.3%      | XLM-R |
| eval_f1                  | 0.908196      | <span style='background-color:green; color:white'>0.919434</span>     | +1.2%      | XLM-R |
| eval_accuracy            | 0.986812      | <span style='background-color:green; color:white'>0.989860</span>     | +0.3%      | XLM-R |
| eval_runtime (s)         | <span style='background-color:green; color:white'>12.321300</span> | 20.287100    | +64.7% | mBERT |
| eval_samples_per_second  | <span style='background-color:green; color:white'>269.615000</span> | 163.749000   | -39.3%     | mBERT |
| eval_steps_per_second    | <span style='background-color:green; color:white'>16.881000</span> | 10.253000    | -39.3%     | mBERT |
| epoch                    | 3.000000      | 3.000000     | 0%             | – |

<span style='text-align:center'><b>Table 1</b>. Comparison between m-BERT and XLM-R models in terms of performance and efficiency.</span>

$$\text{\delta}=\frac{(\text{XLMR-MBERT})}{\text{MBERT}}\times 100$$

In spite of lower in efficiency compared to m-BERT, XLM-R is the better long-term solution for production. While m-BERT is faster and cheaper to serve, the performance improvements of XLM-R provides better results (which are good for production-grade model). Equally important, XLM-R was pretrained on larger and more balanced multilingual corpus. This factor makes it better for crosslingual transfer and better solution to handle future multilingual expansion later. Even though XLM-R is potentially more expensive in its serving, scalability and generalization factors still outweigh the computational cost, particularly related to execution time.

The computational cost, if not possible to keep it minimum with the optimization on the ML side, can still be done by maximizing on the deployment infrastructure such as batching and managing cold starts is still possible. Putting a reasonable amount of data into a single request (or batch) will be more cost-efficient than processing a single data at a time sequentially since GPUs can perform the same operation on multiple data points simultaneously. The parallel processing reduces the billable execution time for on the GPUs.


>[!important]
> To read how the experiment was carried out, use this hyperlink: 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZeKSWbM1zbFjSTrOTAEr8SIXipfpKm23#scrollTo=FXB-7ILRg7rQ)

## Feedback
If there are any questions or suggestions for improvements, feel free to contact me here:

<a href="https://www.linkedin.com/in/adelia-januarto/" target="_blank">
    <img src="https://raw.githubusercontent.com/maurodesouza/profile-readme-generator/master/src/assets/icons/social/linkedin/default.svg" width="52" height="40" alt="linkedin logo"/>
  </a>
<a href="mailto:januartoadelia@gmail.com" target="_blank">
    <img src="https://raw.githubusercontent.com/maurodesouza/profile-readme-generator/master/src/assets/icons/social/gmail/default.svg"  width="52" height="40" alt="gmail logo"/>
  </a>