# Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].

Can big models "feel each other's hearts"? StateBridge achieves multi-agent latent communication with zero training. Introduction: Multi-agent systems rely on textual communication, akin to transmitting a "symphony" through "telegraphs," inevitably losing information in the process of digitization. StateBridge proposes a novel approach - allowing agents to directly exchange continuous hidden states, and translate the sender's "inner world" into language that the receiver can understand through a closed form of orthogonal alignment, without any training or alteration to the model structure. In 26 model-task combinations and 8 benchmark tests, it achieved optimal or runner-up results in 22 out of 26 cases, with an average score that is the highest across all model settings. Paper information project
Content
Title
StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems  
Author
Yanwen Peng, Delvin Ce Zhang, Xi Wang, Nikolaos Aletras  
Unit
University of Sheffield, Computer Science Department, UK
Publish
COLM 2026 (Language Modeling Conference)
I. Research Background: The Communication Bottleneck in Multi-Agent Systems
1.2 Information Loss in Textual Communication However, the current communication methods in multi-agent systems present a fundamental bottleneck: agents can only communicate through text (discrete words). After the sender's intelligent agent has completed its thinking, its internal state is first mapped into discrete lexical units, which are then reconstructed by the receiver's intelligent agent into its own internal representation. This "encoding-decoding" process means that any information that cannot be expressed in terms of lexical identity is discarded at the moment of discreteness.
In a nutshell, when a large model generates each word, its internal hidden state contains far more information than the word itself - continuity of meaning, confidence in inference, and even multiple possibilities for "the next word." Forcing this continuous information into discrete symbols is like transmitting an orchestra through telegraphs: the notes are received, but the melody, intensity, and harmony have long since been lost.
1.3 Why "Direct Transmission" is Unworkable: A naive idea is that since the hidden state information is rich, why not simply pass the sender's hidden state to the receiver directly? However, experiments have shown that this approach is not feasible. The reason is that the hidden state, although correctly dimensioned, is located in a representational space completely different from the input embedding space familiar to the recipient during pre-training. Dimensions are the same, but they do not equal semantic readability. This is like two people using the same 26 letters, but one writing in English and the other in French—by simply swapping the letter sequences, they cannot convey any valid information.
II. Relevant Work: Limitations of Existing Solutions In response to the aforementioned issues, there are some exploratory solutions in academia, each with its own shortcomings. Comparative analysis can be made from four dimensions:
Method category
Is training required?
Model modification
Message injection location
Message storage overhead
Text Communication (TextMAS)
No
None
Discrete lexical elements
—  
Layer-by-layer injection (KV cache transmission, LatentMAS)
No
Each layer injection
All L-layer
O(TLd)  
Learning projector
Yes
Input embedding layer
O(Kd)  
StateBridge (this article) does not have an input embedding layer O(Kd)
2.1 Text communication: Simple but information-lossy is the mainstream solution. Agents exchange information through natural language, which has the advantages of simplicity, generality, and interpretability, but comes at the cost of the information discretization loss mentioned above—which constitutes the implicit "ceiling" of multi-agent system capabilities.
2.2 Latent injection (KV cache transmission): A powerful "hard" solution represented by LatentMAS, injects messages into the transformer layers of the receiver with key-value caching. This method does not require training, but has two major issues. First, storage overhead is huge: it requires storing complete key-value states for each layer, with message storage complexity O(TLd), i.e., proportional to the product of sequence length, number of layers, and dimensionality. Second, portability is poor: latent injection requires that both sender and receiver have exactly the same layer structure; once the layer structures differ between model families, the method fails—this is fully confirmed in the cross-family experiments later on.
2.3 Learning-based projectors: Flexible but costly Another approach is to train a projector to map hidden states to the input embedding space. These methods are flexible but require additional training data and computational power, contradicting the "plug-and-play" lightweight deployment requirements and also difficult to reuse continuously in the context of rapid model iterations.
III. Method: StateBridge — A "Zero Training" Semantic Bridge
The core insight of StateBridge can be summarized in one sentence: solve cross-space communication problems without necessarily "learning" a mapping, but rather "solving" one. By performing a set of pure algebraic operations to achieve geometric alignment between the hidden state space and the input embedding space in a closed form —without any training, no learnable parameters introduced, no change to the transformer structure.
3.1 Four-Agent pipeline StateBridge adopts a pipeline architecture with four roles working collaboratively:
Planner → Critic → Refiner → Judger Each intelligent entity completes its segment, and its "message" is no longer transmitted in text form but as a continuous hidden state, aligned and injected as a continuous prefix into the input of the next intelligent entity. It is worth noting that the injection position is carefully chosen at the input embedding layer—the minimal invasion point of the model structure and the key source of portability.
3.2 Five-step Alignment Process In each pair of adjacent intelligent entities, StateBridge performs five steps:
First step: Extraction of message states. The sender intelligent entity records the hidden state outputted during its generation process, along with the reference embedding corresponding to the decoding tokens, serving as anchor points for alignment.
Second step: Normalization of two spaces. Center and whiten the distribution of hidden states and reference embeddings separately, eliminating mean shift and anisotropy differences, making the two distributions comparable in coordinates.
Third step: Align geometric structures. Solve a Procrustes problem to obtain a closed-form optimal orthogonal transformation through singular value decomposition (SVD). Since the solution is orthogonal, the alignment process preserves distance and angle—the distance and angle relationship between the hidden states of the sender are fully preserved after the transformation.
Fourth step: Calibration of compatibility. Restoring statistical features of the input space, align the norm distribution of word table embeddings, and "softly anchor" each state towards neighboring word table embedding directions, ensuring numerical scale compatibility between the prefix and the actual input embedding.
Fifth step: Injection of continuous prefixes. Concatenate the aligned state as a continuous prefix at the beginning of the receiver's input. The receiver will assign sequentially increasing position numbers to these prefixes, and their attention mechanism and position encoding will treat these prefixes like ordinary word embeddings—to the downstream model, this "seems" to be just a normal input.
3.3 Why orthogonal alignment is so crucial: A compelling answer to the question of why orthogonal Procrustes is replaced by Ridge Regression in ablations: While pointwise reconstruction error is lower, the average score drops from 82.4 to 74.9. The reason is that the Ridge Regression aims to minimize the pointwise error, which distorts the pairwise geometric relationship between states—the very geometric relationship that carries semantic similarity information. The properties of orthogonal transformations that preserve distance and angle, precisely safeguard the integrity of semantic structures. This suggests that the goal of alignment is not "point-by-point approximation" but "structural fidelity".
3.4 The whitening and anchoring computation complexity during the alignment process are O(d³) and O(KVd), respectively, with each batch requiring only one calculation, resulting in overall overhead lower than that of a single generation process. State extraction requires only recording the final layer output, and message storage has an overhead of O(Kd), significantly reducing compared to O(TLd) for layer-by-layer injection—making potential communication truly "affordable" in practice.
IV. Experimental Setup: Strict and Fair Testing 4.1 Benchmarks and Task Experiments Cover 8 benchmark tests, three major task categories.
•  
Mathematical reasoning: GSM8K, AIME 2024, AIME 2025
•  
Question-Answering Comprehension: GPQA-Diamond, ARC-Challenge, MedQA
•  
Code generation: MBPP+, HumanEval+
Among them, GPQA-Diamond is a graduate-level "anti-search" question benchmark (198 questions), MedQA originates from the United States Medical Licensing Examination (assessing subset of 300 questions), and AIME is the American Mathematical Olympiad - all are recognized as high difficulty challenging tasks.
4.2 Model Setups: Verification under four model setups: Qwen3-4B, OLMo3-7B-Think, Qwen3-8B, Qwen3-32B. Notably, OLMo3-7B-Think exhibits significant differences in layer structure compared to the Qwen family, providing a natural testbed for examining the portability of methods across model families.
4.3 Fair Uniform Configuration: To ensure the conclusions are trustworthy, all hyperparameters were fine-tuned only once in one setting (Qwen3-4B + MedQA) and then applied unchanged to all datasets and models without any targeted adjustments. Reported accuracy for question answering and mathematical reasoning tasks, code generation task reported pass@1. The experiments were conducted on two NVIDIA A100-80G machines with sampling temperature at 0.6 and top-p at 0.95.
V. Experimental Results: Comprehensive and robust leading results across 26 model-task evaluations, StateBridge achieved optimal or runner-up performance in 22 out of 26 cases, with the highest average score across all model settings:
Model Settings
StateBridge
Strongest Baseline
Improvement Ratio
Optimal/Runner-up Number
Qwen3-4B
82.480.0 (LatentMAS)
+2.4
5 / 5
OLMo3-7B-Think
76.773.9 (TextMAS)
+2.8
4 / 5
Qwen3-8B
74.371.8 (LatentMAS)
+2.5
7 / 8
Qwen3-32B
81.078.1 (TextMAS)
+2.9
6 / 8
5.2 The advantages are concentrated on "hard nuts" It is worth noting that the enhancement of StateBridge was not derived from simple task "boosting", but rather highly focused on the most challenging benchmarks: in Qwen3-8B, GPQA improved by +7.0, AIME 2024 improved by +6.6, and MedQA improved by +4.0. The more difficult the task, the more deadly the information loss, and the greater the potential value of communication - this principle perfectly aligns with the theoretical motivation of the method, also indicating that the gain is not due to noise-like random perturbations.
5.3 Cross-family portability verification In OLMo3-7B-Think, the average score of the layer-by-layer injection scheme (KV cache transmission) was only 55.1%, even lower than the pure text communication rate of 73.9% - this indicates that when the model family layer structure is different, layer-by-layer injection will "fit like a glove". However, StateBridge achieved 76.7%, significantly better than both baselines. This confirms the portability advantage brought about by the design decision of "operating only on the input embedding layer": it does not rely on any internal structure specific to any model family.
5.4 Ablation experiments: Each component has its value Based on the average score of Qwen3-4B, the ablation experiments clearly demonstrate the contribution of each component:
Configuration  
Average Score  
Complete StateBridge  
82.4 Orthogonal Procrustes substitution for Ridge Regression  
74.9  
Removing norm regularization  
79.5  
Removing word anchoring  
80.2  
Random noise prefix (control)  
48.8  
It is particularly noteworthy that the control group: merely adding "continuous vectors" did not bring benefits - the performance of the random noise prefix dropped sharply to 48.8, eliminating the possibility that "gains were solely due to additional continuous vectors", proving that the performance improvement indeed came from alignment itself, not from the operation form of injection.
5.5 Qualitative Analysis: The study team on the prefix-carrying information "exceeds its lexical source" also designed a clever qualitative experiment: in one case of MedQA, the critiquer was asked to reconstruct the planner's plan solely based on aligned prefixes. When the length of the prefix was only K=16, the visible text suffixes were fragmented, yet the critiquer still successfully restored the diagnosis conclusion, key clinical features, and exclusion of alternatives, even including a term never seen in the visible lexicon, "koilonychia" (reverse nail).
This constitutes a compelling evidence: the information carried by aligned hidden prefixes of post-alignment state exceeds the scope that it originates from. In other words, potential communication is not only "losslessly conveying text", but also "transmitting semantics beyond the text".
VI. Discussion: Why "StateBridge" effectively addresses zero-training needs better than "learning". The clever aspect of StateBridge is that it redefines "representing spatial alignment" as a purely geometric problem. The hidden state and input embedding have the same dimension, but what's missing is the correspondence between positions and coordinate systems. The orthogonal Procrustes provides the optimal rotation transformation in a closed form, ensuring both mathematical optimality (minimizing the Euclidean norm error under orthogonal constraints) and inherent geometric properties of preserving distance and angle. The mathematical structure itself, replacing the data-driven learning process - that's exactly what makes "zero training" possible.
6.2 Information Preservation and Information Carryover From text to lemmas, then from lemmas to states, each step involves information compression. StateBridge bypasses the first compression - allowing messages to flow directly in a continuous form. The collaborative design of five components (centralization, bleaching, orthogonal alignment, norm calibration, and lexical anchoring) addresses five major issues: coordinate system offset, anisotropy, rotation alignment, numerical compatibility, and lexical affinity respectively. Each component is essential for the success of the experiment, as evidenced by the data collected through the ablation study.
6.3 Limitations and Prospects
It must be candidly stated that StateBridge currently targets homogeneous multi-agent systems—where all agents share the same pre-trained weights. The challenge of aligning heterogeneous sender-receiver pairs, where the sender and receiver use different models, is a more severe test for alignment methods and represents the direction of future work explicitly identified by the author. Additionally, how to further reduce the length of prefixes and how to maintain alignment stability in longer contexts are also areas worthy of subsequent exploration.

VII. Conclusion
StateBridge proposes a zero-training, structure-agnostic, single-point injection method for latent communication: it achieves continuous message transmission between large model agents by completing hidden state orthogonal alignment at the input embedding layer in a closed form. In the 26 model-task evaluations, it outperforms textual communication and layer-wise injection baselines with 22 out of 26 optimal/tie-for-best scores, achieving the highest average score across all model settings and demonstrating unique portability advantages in cross-model family scenarios.
The profound implication of this work might lie in the fact that "translation" between agents need not necessarily involve learning when the representation space is sufficiently regularized; instead, it may only require an appropriate rotation. On the path towards stronger multi-agent collaboration systems, this undoubtedly marks a milestone worth noting.
References
1. Peng Y, Zhang D C, Wang X, et al. StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems[C]. Conference on Language Modeling (COLM), 2026.
Zou J, Yang X, Qiu R, et al. Latent Collaboration in Multi-Agent Systems[J]. arXiv preprint arXiv:2511.20639, 2025.  
3  
Rein D, Hou B L, Stickland A C, et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark[C]. First Conference on Language Modeling, 2024.  
4  
Jin D, Pan E, Oufattole N, et al. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams[J]. arXiv preprint arXiv:2009.13081, 2020.  

## Images

![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](01.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](02.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](03.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](04.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](05.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](06.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](07.jpg)
![Can EEG be used between large models? StateBridge achieves covert communication among multiple agents with zero training [with Python code].](08.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
