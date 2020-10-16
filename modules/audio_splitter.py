from pydub import AudioSegment
import os
import math


class AudioSplitter:
    """

    """
    def __init__(self, audio_folder_path, audio_folder_output):
        self.audio_folder_path = audio_folder_path
        self.audio_folder_output = audio_folder_output
        self.__create_if_does_not_exist(audio_folder_output)

    def get_all_file_paths(self, path):
        result_file_paths = []

        for inst in os.listdir(path):
            recursive_file_instances = []
            if os.path.isdir("{}/{}".format(path, inst)):
                recursive_file_instances = self.get_all_file_paths("{}/{}".format(path, inst))
                for file_path in recursive_file_instances:
                    result_file_paths.append(file_path)
            else:
                result_file_paths.append("{}/{}".format(path, inst))

        return result_file_paths + recursive_file_instances

    def split_file_in_multiple_files(self, file_path, seconds_for_split):
        file_name = file_path.split('/')[-1]
        audio = AudioSegment.from_wav(file_path)
        number_of_splits = math.ceil(audio.duration_seconds / seconds_for_split)

        for i in range(0, number_of_splits):
            new_split = audio[i * (seconds_for_split * 1000):(i + 1) * (seconds_for_split * 1000)]
            new_file_name = self.audio_folder_output + file_name + "-split-" + str(i + 1) + ".wav"
            new_split.export(new_file_name, format="wav")  # Exports to a wav file in the current path.

    def do_audio_splitting(self, seconds_for_split):
        for file_path in self.get_all_file_paths(self.audio_folder_path):
            self.split_file_in_multiple_files(file_path, seconds_for_split)

    @staticmethod
    def __create_if_does_not_exist(audio_folder_output):
        if not os.path.exists(audio_folder_output):
            os.makedirs(audio_folder_output)
