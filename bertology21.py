import numpy as np
import scipy.optimize


papers = {
    0: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    1: 'taxonomy',
    2: 'A Structural Probe for Finding Syntax in Word Representations',
    3: 'What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models',
    4: 'Do NLP Models Know Numbers? Probing Numeracy in Embeddings',
    5: 'What’s in a Name? Are BERT Named Entity Representations just as Good for any other Name?',
    6: 'Language Models as Knowledge Bases?',
    7: 'Linguistic Knowledge and Transferability of Contextual Representations',
    8: 'Revealing the Dark Secrets of BERT',
    9: 'Information-Theoretic Probing with Minimum Description Length',
    10: 'Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned',
    11: 'Quantity doesn’t buy quality syntax with neural language models',
    12: 'Energy and Policy Considerations for Deep Learning in NLP',
    13: 'Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data',
    14: 'Can a Fruit Fly Learn Word Embeddings?'
}

''' the choice of 1st paper will be outweighed by two people getting at least one item of interest '''
max_weight = 7

students = {
    0: ['Annegret Janzso', '2568385', 's8anjanz@stud.uni-saarland.de', [13, 0, 14, 7]],
    1: ['Peilu Lin', '7010601', 'peli00002@stud.uni-saarland.de', [6, 8, 12]],
    2: ['Meng Li', '7010888', 'meli00001@stud.uni-saarland.de', [7, 6, 13]],
    3: ['Leonie Harter', '?', 'leonie-harter@web.de', [5, 2, 4]],
    4: ['Rricha Jalota', '7010592', 'rrja00001@stud.uni-saarland.de', [14, 8, 13, 6]],
    5: ['Katharina Stein', '2563513', 'kstein@coli.uni-saarland.de', [3, 6, 5]],
    6: ['Sangeet Sagar', '7009050', 'sasa00001@stud.uni-saarland.de', [8, 10, 6]],
    7: ['Pavle Markovic', '7007913', 'markovic.pavle@outlook.com', [9, 7, 10, 8]],
    8: ['Axel Allén', '2579024', 's8axalle@stud.uni-saarland.de', [8, 10, 3, 12, 14]],
    9: ['Pin-Jie Lin', '7010904', 'pjlintw@gmail.com', [0, 10, 2, 6]],
    10: ['Suruthai Noon Lywen Pokaratsiri Goldstein', '2581469', 'supoka@coli.uni-saarland.de', [8, 3, 6, 13]]
}

emails = ';'.join([students[ind][-2] for ind in students])
# s8anjanz@stud.uni-saarland.de;peli00002@stud.uni-saarland.de;meli00001@stud.uni-saarland.de;leonie-harter@web.de;rrja00001@stud.uni-saarland.de;kstein@coli.uni-saarland.de;sasa00001@stud.uni-saarland.de;markovic.pavle@outlook.com;s8axalle@stud.uni-saarland.de;pjlintw@gmail.com;supoka@coli.uni-saarland.de

weight_matrix = np.zeros(shape=[11, 15])
for student_index, student_info in students.items():
    for paper_index_index, paper_index in enumerate(student_info[-1]):
        weight_matrix[student_index][paper_index] = max_weight - paper_index_index

x, y = scipy.optimize.linear_sum_assignment(-weight_matrix)
for index in range(10):
    print(f'{students[int(x[index])][0]}: {papers[int(y[index])]}')

# Annegret Janzso: Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data
# Peilu Lin: Language Models as Knowledge Bases?
# Meng Li: Linguistic Knowledge and Transferability of Contextual Representations
# Leonie Harter: What’s in a Name? Are BERT Named Entity Representations just as Good for any other Name?
# Rricha Jalota: Can a Fruit Fly Learn Word Embeddings?
# Katharina Stein: What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models
# Sangeet Sagar: Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned
# Pavle Markovic: Information-Theoretic Probing with Minimum Description Length
# Axel Allén: Energy and Policy Considerations for Deep Learning in NLP
# Pin-Jie Lin: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
