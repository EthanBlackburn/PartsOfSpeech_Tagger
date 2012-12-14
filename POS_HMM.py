
#for this hidden markov model, the words are the observations and the tags are the hidden states.
start_prob = {}
transition_prob = {}
emission_prob = {}

#The states are the word tags. Credits for the tags goes to NYU
States = ['CC','CD','DT','EX','FW','IN','HYPH','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PP'
,'PP$','PRP','PRP$','RB','RBR','RBS','RP','.','\'\'',':','\`\`','COMMA','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

def flatten_dict_values(d):
    values = []
    for value in d.itervalues():
        if isinstance(value, dict):
            values.extend(flatten_dict_values(value))
        else:
            values.append(value)
    return values
	
#initialize transition and start probabilities
for i in range(len(States)):
	transition_prob[States[i]] = {}
	start_prob[States[i]] = 0
	emission_prob[States[i]] = {}
	for j in range(len(States)):
		transition_prob[States[i]][States[j]] = 0

#train on training set

def Train():
	path = raw_input('enter path to training file: ')
	f = open(path,'r')
	new_sentence = True
	start_count = 0.0
	str_previous = ''
	for line in f.readlines():
		if(line.strip() != ''):
			fixed_line = line.replace(' ', '\t')
			str_data = fixed_line.split('\t')
			tag = str_data[1].strip('\r\n')
			if(States.count(tag) == 0):
				tag = 'SYM'
			if(new_sentence == True):
				start_count+=1
				start_prob[tag] +=1
				str_previous = tag
				new_sentence = False
			elif(tag == '.'):
				new_sentence = True
				transition_prob[str_previous]['.'] += 1
				str_previous = ''
			else:
				if(transition_prob[str_previous].has_key(tag) == False):
					transition_prob[str_previous]['SYM'] += 1
					str_previous = 'SYM'
				else:
					transition_prob[str_previous][tag] += 1
					str_previous = tag
			emission_prob[tag].setdefault(str_data[0],0)
			emission_prob[tag][str_data[0]] +=1
		
		else:
			new_sentence = True
	f.close()
		
	#calculate emission, transition and start probabilities
	for i in range(len(States)):
		count = 0.0
		for j in range(len(States)):
			count += transition_prob[States[i]][States[j]]
			if(start_count >1):
				start_prob[States[i]] /= start_count
		for j in range(len(States)):
			if(count > 0):
				transition_prob[States[i]][States[j]] /= count
			else:
				break
	return 0
			
def Viterbi(O, states, start, trans, emission):
	V = [{}]
	path = {}
	final_path = ''
	final_tag = ''
	#start probabilities and path
	for tag in states:
		try:
			V[0][tag] = start[tag]*emission[tag][O[0]]
			path[tag] = tag
		except KeyError:
			try:
				V[0][tag] = start[tag]*emission[tag][O[0].lower()] #check if it has a known lowercase tag
				path[tag] = tag
			except KeyError:
				try:
					V[0][tag] = start[tag]*emission[tag][O[0][0:1].upper()+O[0][1:-1]] #check if it has a known upper case tag
					path[tag] = tag
				except KeyError:
					V[0][tag] = 0
	#find probability for each observation
	possible_tags = {}
	for i in range(1,len(O)):
		V.append({})
		#probability of current being in the state "tag1"
		max_tag = ''
		max_prob  = 0.0
		for tag1 in states:
			path_probs = []
			for tag2 in states:
				try:
					path_probs.append(V[i-1][tag2]*emission[tag1][O[i]]*trans[tag2][tag1])
				except KeyError:
					try:
						path_probs.append(V[i-1][tag2]*emission[tag1][O[i].lower()]*trans[tag2][tag1]) #check if it has a known lowercase tag
					except KeyError:
						try:
							path_probs.append(V[i-1][tag2]*emission[tag1][O[i][0:1].upper()+O[i][1:-1]]*trans[tag2][tag1]) #check if it has a known uppercase tag
						except KeyError:
							path_probs.append(0)
				if(path_probs[-1] > max_prob):
					max_tag = tag2
					max_prob = path_probs[-1]

			#Set the Viterbi for this observation and tag to the maximum of the path probabilities	
			V[i][tag1] = max(path_probs)
			if(i == len(O)-1):
				if(final_tag == ''):
					final_tag = tag1
				elif(V[i][tag1] > V[i][final_tag]):
					final_tag = tag1
					
		#if tag was not found above, we find its greatest emission value and substitute that because it is the most likely state
		if(max_tag == ''):
			max_emission = 0.0
			for t in range(len(states)):
				try:
					if(emission[states[t]][O[i]] > max_emission):
						max_emission = emission[states[t]][O[i]]
						max_tag = states[t]
				except KeyError:
					try:
						if(emission[states[t]][O[i].lower()] > max_emission):
							max_emission = emission[states[t]][O[i].lower()]
							max_tag = states[t]
					except KeyError:
						try:
							if(emission[states[t]][O[i][0:1].upper()+O[i][1:-1]] > max_emission):
								max_emission = emission[states[t]][O[i][0:1].upper()+O[i][1:-1]]
								max_tag = states[t]
						except KeyError:
							continue
		if(final_path != ''):
			final_path = final_path + "-->" + max_tag
		else:
			final_path = max_tag
		max_tag = ''
		max_prob = 0.0
				
	final_path = final_path + "-->" + final_tag
	
	return final_path

def Test():
	f3_path = raw_input('Enter path of file you would like to the model on: ')
	f3 = open(f3_path,'r')
	tag = {}
	observation = {}
	prediction = []
	count = 0
	for line in f3.readlines():
		if(line.strip() != ''):
			fixed_line = line.replace(' ', '\t')
			str_data = fixed_line.split('\t')
			tag[count] = str_data[1].strip('\r\n')
			observation[count] = str_data[0]
			if(States.count(tag[count]) == 0):
				tag[count] = 'SYM'
			if(tag[count] == '.'):
				print "Model predicts tag sequence to be: " + Viterbi(observation, States, start_prob, transition_prob,emission_prob)
				print "Actual tags sequence is: " + '-->'.join(flatten_dict_values(tag))
				predicted = Viterbi(observation, States, start_prob, transition_prob,emission_prob).split('-->')
				for t in range(count):
					prediction.append(predicted[t] == tag[t])
				count = 0
				tag.clear()
				observation.clear()
			else:
				count += 1
		else:
			count = 0

	correct_count = 0.0				
	for i in range(len(prediction)):
		if(prediction[i]):
			correct_count += 1
		
	print "Accuracy of model: " + str(correct_count/float(len(prediction)))
	
	return 0
	
training_sets = input('how many training set would you like to run on? ')

for i in range(training_sets):
	Train()
	
test_sets = input('how many files would you like to test? ')

for i in range(test_sets):
	Test()
	




			
			
		
		
			
		
			
			


