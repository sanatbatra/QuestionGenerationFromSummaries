import csv
file = open("filtered_test_set.txt","r")
text = file.read()
text_split = text.split('<|endoftext|>')
csv_input = []
text_split = text_split[:-1]
for s in text_split:
    s_split = s.split('[question]')
    csv_input.append({'prompt': '[summary]' + s_split[0].split('[summary]')[1] + '[question]',
                      'gold_standard_question': s_split[1][:-1], 'generated_question': ""})

print(len(csv_input))
with open('filtered_test_set.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, ['prompt', 'gold_standard_question', 'generated_question'])
    dict_writer.writeheader()
    dict_writer.writerows(csv_input)
