import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from external.vqa.vqa import VQA


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            word_list = self._create_word_list([d['question'] for d in self._vqa.questions['questions']])
            self.question_word_to_id_map = self._create_id_map(word_list, question_word_list_length)
            ############
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            answer_list = []
            # Create answer list
            for dct in self._vqa.qa.values():
                for answers in dct['answers']:
                    answer_list.append(answers['answer'])
            self.answer_to_id_map = self._create_id_map(answer_list, answer_list_length)
            ############
        else:
            self.answer_to_id_map = answer_to_id_map


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        word_list = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = ''.join([i for i in sentence if (i.isalpha() or i == ' ')])
            words = sentence.split()
            for word in words:
                word_list.append(word)
        return word_list


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO
        return_dict = {}
        for word in word_list:
            try:
                return_dict[word] += 1
            except:
                return_dict[word] = 1
        sorted_list = sorted(return_dict.items(), key=lambda k:k[1], reverse=True)
        return_dict.clear()
        for i in range(max_list_length if len(sorted_list) > max_list_length else len(sorted_list)):
            return_dict[sorted_list[i][0]] = i
        return return_dict
        ############
        

    def __len__(self):
        ############ 1.8 TODO
        # Length of all questions
        return len(self._vqa.questions['questions'])
        ############

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        # print("Loading IDs and qa")
        question = self._vqa.questions['questions'][idx]['question']
        question_id = self._vqa.questions['questions'][idx]['question_id']
        image_id = self._vqa.questions['questions'][idx]['image_id']
        image_id = f'{image_id:012}'
        answers = self._vqa.qa[question_id]['answers']
        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            # the caching and loading logic here
            feat_path = os.path.join(self._cache_location, f'{image_id}.pt')
            try:
                image = torch.load(feat_path)
            except:
                image_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
                image = Image.open(image_path).convert('RGB')
                image = self._transform(image).unsqueeze(0)
                image = self._pre_encoder(image)[0]
                torch.save(image, feat_path)
        else:
            ############ 1.9 TODO
            # print("Get image")
            # load the image from disk, apply self._transform (if not None)
            img_path = os.path.join(
                    self._image_dir, self._image_filename_pattern.format(image_id))
            image = Image.open(img_path).convert('RGB')
            # print("Apply image transforms")
            if self._transform is not None:
                image = self._transform(image)
            ############

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        # print("Encoding questions")
        question_tensor = torch.zeros((self._max_question_length, self.question_word_list_length))
        # Split up question
        sentence = question.lower()
        sentence = ''.join([i for i in sentence if (i.isalpha() or i == ' ')])
        words = sentence.split()
        for i in range(self._max_question_length):
            if i < len(words):
                try:
                    question_tensor[i, self.question_word_to_id_map[words[i]]] = 1
                except:
                    question_tensor[i, self.unknown_question_word_index] = 1
        
        answers_tensor = torch.zeros((10, self.answer_list_length))
        for i in range(10):
            try:
                # For all answers within the dictionary
                answers_tensor[i, self.answer_to_id_map[answers[i]['answer']]] = 1
            except:
                # For answers not in answer list, use last column
                answers_tensor[i, self.unknown_answer_index] = 1
        ############
        return {
            'idx': idx,
            'image': image,
            'question': question_tensor,
            'answers': answers_tensor
        }
