# from google.colab import drive
from IPython.display import Audio
# drive.mount('/content/drive')
# !pip install koreanize_matplotlib
# import koreanize_matplotlib
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import librosa
import matplotlib .pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import wave
import scipy.signal

model = hub.load('https://tfhub.dev/google/yamnet/1')
audio_total_time = ""

numbering = {
    "이갈이" :0,
    "기침" :1,
    "코골이" :2,
    "잠꼬대" :3
}


# 가중치= 전체 데이터 개수/ 총등장횟수

 
bruxism_score_dic = {
    0:2.94,
    1:1.76,
    2:1.76,
    3:1.47,
    4:0.29,
    5:0.88,
    6:0.01,
    7:0.29,
    8:0.29,
    9:0.01,
    10:0.29,
    11:0.01,
    12:0.01,
    
}
snoring_score_dic = {
    0:8.67,
    1:0.01,
    2:1.33
}

cough_score_dic = {
    #기침와 이갈이는 그룹조합의 인덱스번호 말고도 조건이 하나 더있기떄문에 특별 점수가 하나더 존재해야한다.(only_cough에대한 score값 저장 )
    0 : 1.5,
    "only_cough" : 1.0
}
speech_score_dic = {
    0:5,
    1:1.0,
    2:1.0,
    3:1.5,
    4:1.0,
    5:1.2,
    6:1.0,
}
score_dic_matching = {
    "이갈이" :bruxism_score_dic,
    "기침" :cough_score_dic,
    "잠꼬대" :speech_score_dic,
    "코골이" :snoring_score_dic
}
def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    # print(f"{original_sample_rate} 을 {desired_sample_rate}으로변환합니다")
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
def get_class_name(class_id):
    """Returns class name for a given class_id."""
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    else:
        return None  # class_id가 범위를 벗어나면 None을 반환
group_size_dic = {1: 0.96, 2: 1.44, 3: 1.92, 4: 2.4, 5: 2.88, 6: 3.36, 7: 3.84, 8: 4.32, 9: 4.8, 10: 5.28, 11: 5.76, 12: 6.24, 13: 6.72, 14: 7.2, 15: 7.68, 16: 8.16, 17: 8.64, 18: 9.12, 19: 9.6}
total_group_size_calculate_dic   = {1: 40,
 2: 39,
 3: 38,
 4: 37,
 5: 36,
 6: 35,
 7: 34,
 8: 33,
 9: 32,
 10: 31,
 11: 30,
 12: 29,
 13: 28,
 14: 27,
 15: 26,
 16: 25,
 17: 24,
 18: 23,
 19: 22}

global_class_combinations_list = [
    [
    # 동물 소리 포함 이갈이
    {500 : (0.1,0.8), 41: (0.05, 0.5) }, #41 Snort
    {369:(0.1,0.5),500:(0.1,0.5)}, #369 Toothbrush
    {412:(0.05,0.4),431:(0.05,0.4),470:(0.05,0.4)}, #412 : Tools , 431: Wood , 470: Rub
    {127: (0.1, 0.8), 67: (0.1, 0.7), 103: (0.1, 0.6)},  # Frog, Animal, Wild animals
    # 파열음 포함 이갈이
    {399: (0.1, 0.9), 403: (0.1, 0.8), 412: (0.1, 0.7)},  # Ratchet, pawl, Gears, Tools

    # 카메라 소리 포함 이갈이
    {410: (0.1, 0.9), 411: (0.1, 0.8), 398: (0.1, 0.7)},  # Camera, Single-lens reflex camera, Mechanisms

    # 그르륵 칵칵 소리 포함 이갈이
    {412: (0.1, 0.8), 410: (0.1, 0.7), 398: (0.1, 0.6)},  # Tools, Camera, Mechanisms

    # 긁는 소리 포함 이갈이
    {435: (0.1, 0.9), 438: (0.1, 0.8), 449: (0.1, 0.7)},  # Glass, Liquid, Stir

    # 동전 소리 포함 이갈이
    {374: (0.1, 0.9), 436: (0.1, 0.8), 435: (0.1, 0.7)},  # Coin dropping, Chink, clink, Glass

    # 지퍼 소리 포함 이갈이
    {372: (0.1, 0.9), 434: (0.1, 0.8), 469: (0.1, 0.7)},  # Zipper, Crack, Scrape

    # 창문 닦는 소리 포함 이갈이
    # {500: (0.1, 0.8), 439: (0.1, 0.7), 50: (0.1, 0.6)},  # Inside small room, Splash, Biting

    # 박수치는 소리 포함 이갈이
    {399: (0.1, 0.9), 403: (0.1, 0.8), 410: (0.1, 0.7)},  # Ratchet, pawl, Gears, Camera

    # 과자 씹는 소리 포함 이갈이
    {410: (0.1, 0.9), 398: (0.1, 0.8), 435: (0.1, 0.7)}   # Camera, Mechanisms, Glass
    ],

    #기침 조합입니다
    [
    {42: (0.1, 1.0)},  # 기침 단독
    ],
    #코골이 조합입니다.
    [
    {38: (0.8, 1.0), 36: (0.2, 0.6)},  # Snoring과 함께 일정 수준의 숨소리가 포함되어야 함
    {38: (0.9, 1.0), 23: (0.2, 0.8), 36: (0.2, 0.5)},  # Snoring + Sigh + Breathing 조합
    {38: (0.8, 1.0), 0: (0.0, 0.5)},  # Speech가 높으면 Snoring으로 판단하지 않음
    ],
    #잠꼬대 조합입니다.
    [
    {0: (0.6, 1.0)},  # Speech 비율이 높은 경우
    {0: (0.5, 0.7), 2: (0.3, 0.7)},  # 음성 + 대화(Conversation)
    {0: (0.5, 0.7), 3: (0.2, 0.5)},  #음성 + 내레이션(Narration)
    {0: (0.5, 0.7), 5: (0.2, 0.4)},  #음성 + 음성 합성기(Speech synthesizer)
    {0: (0.5, 0.7), 132: (0.2, 0.5)},  #음성 + 음악(Music)
    {0: (0.5, 0.7), 36: (0.1, 0.3)},  # 음성 + 숨소리(Breathing)
    ]
]


group_size_list = [
    #이갈이 그룹 사이즈 입니다
    4   ,
    #기침 그룹 사이즈 입니다
    3,
    #코골이 그룹 사이즈 입니다
    5,
    #잠꼬대 그룹 사이즈 입니다.
    5,
]
min_valid_count_list = [
    #이갈이 그룹 T/F 판단 최소값 입니다
    2,
    #기침 그룹 T/F 판단 사이즈 입니다
    2,
    #코골이 그룹 T/F 판단 사이즈 입니다
    3,
    #잠꼬대 그룹 T/F 판단 이즈 입니다.
    3,
]
required_true_count_list = [
    #이갈이 최종 판단 사이즈 입니다
    2,
    #기침 최종 판단 사이즈 입니다
    2,
    #코골이 최종 판단 사이즈 입니다
    2,
    #잠꼬대 최종 판단 사이즈 입니다.
    2,
]
def is_cough_in_top_10(scores):
    """ 상위 5개 클래스 중 Cough(42)가 포함되어 있는지 확인 """
    top_10_indices = np.argsort(scores)[::-1][:10]  # 확률값 기준 상위 5개 클래스 추출
    # get_class_name(class_id)
    # for i in top_5_indices:

    #     print(f"기침 상위권 목록 {get_class_name(i)}")
    return 42 in top_10_indices  # Cough(42)가 상위 5개 안에 있는지 확인

def one_frame_judgment_only_cough(scores, class_combinations_list,current=None):
    """ 기존 조합 기반 판단 + Cough(42) 포함 여부 추가 """
    temp1,score_frame = one_frame_judgment(scores, class_combinations_list)
    # print(f"기침에 대해서 temp1 : {temp1} ,score_frame:{score_frame}")
    # top_5_indices = np.argsort(scores)[::-1][:20]  # 확률값 기준 상위 5개 클래스 추출
    # get_class_name(class_id)
    # for i in top_5_indices:
    #     print(f"기침 상위권 목록 {get_class_name(i)}")
    # print("\n\n")
    # 기존 조합 검사
    if temp1:
        # print(f"기침에대해서 temp1이 True일떄 ,score_frame:{score_frame}")
        return True,score_frame # 기존 조합이 충족되면 바로 True

    temp2 = is_cough_in_top_10(scores)  # Cough(42) 상위 5개 내 포함 여부 검사
    if temp2:
        # print(f"기침에대해서 temp2 True일떄 ,cough_score_dic:{cough_score_dic['only_cough']}")
        return True,cough_score_dic['only_cough']  # 기침이 상위 5개 안에 있으면 True
    # print(f"기침에대해서 temp2 False일떄")
    return False,0  # 둘 다 아니라면 False
# 한 프레임에 대한 T/F 계산 함수 , scores는 yamnet의 한 프레임에 대한 확률벡터값(521,1)
def one_frame_judgment(scores, class_combinations,current=None):
    for i,combination in enumerate(class_combinations):
        all_match = True  # 현재 combination이 완전히 만족하는지 확인하는 변수
        for class_id, score_range in combination.items():

            min_score, max_score = score_range  # 범위로부터 최소값, 최대값 분리
            score = scores[class_id]
            # print(f"{get_class_name(class_id)}의 score값은 현재 : {score}이다")
            if not (min_score <= score <= max_score):  # 하나라도 조건을 만족 못 하면 실패
                all_match = False
                break
        if all_match:  # 해당 combination이 완벽하게 만족하면 True 반환
            if not current:
                return True,0
            # print(f"current : {current}" )
            # print(f"combination : {combination}")
            # print(f"i값 : {i}, 그당시에 score_dic_matching의 current : {score_dic_matching[current]}")
            # print(score_dic_matching[current][i])
            return True,(score_dic_matching[current][i])
    return False,0  # 모든 조합이 실패하면 False 반환


def is_non_bruxism_top_10(scores):
    """상위 10개 클래스 중 Snoring(38), Cough(42), Speech(0)가 포함되어 있는지 확인"""
    top_10_indices = np.argsort(scores)[::-1][:10]  # 확률값 기준 상위 10개 클래스 추출
    non_bruxism_classes = {38, 42, 0}  # 코골이, 기침, 음성 클래스 ID
    # print(f"어케 나왔지 ?? {any(class_id in non_bruxism_classes for class_id in top_10_indices)}")
    return any(class_id in non_bruxism_classes for class_id in top_10_indices)
def one_frame_judgment_only_bruxism(scores, class_combinations_list,current=None):

    match_combination,score_frame = one_frame_judgment(scores, class_combinations_list, current=current)
    contains_non_bruxism = is_non_bruxism_top_10(scores)

    if match_combination and not contains_non_bruxism:
        # print(f"[✅ 이갈이 탐지] 조건 일치 & 방해 클래스 없음 - Frame {frame_idx}")
        return True,score_frame
    # elif match_combination and contains_non_bruxism:
        # print(f"[⚠️ 제외] 조건 일치했지만 방해 클래스 존재 - Frame {frame_idx}")
    # elif not match_combination:
        # print(f"[❌ 불일치] 조합 불일치 - Frame {frame_idx}")

    return False,0


# 오디오 파일에 대한 최종 T/F 판단함수 , audio_path는 오디오 파일, class_combinations 은 조건 , group_size는 내가 묶을 단위의 frame숫자,
# min_valid_count 는 group에서 몇개이상일지 판단하는 값, required_true_count는 실제 최종판단해서 true 갯수 기준 , scores 는 Yamnet의 scores.numpy
def final_logic(class_combinations, group_size, min_valid_count, required_true_count,scores,slide_step=0.48,bruxism=False,cough=False,snoring=False,speech=False):
    real_time = group_size_dic[group_size]  # 오디오 파일 20초로 고정
    result_count = 0
    total_window_size = 40 # 판단해야하는 총 frame수 40으로 고정
    # 구간별 T/F 결과를 딕셔너리 형태로 저장할 리스트 (각 구간에 대한 T/F 값)
    total_group_size  = total_group_size_calculate_dic[group_size] #그룹사이즈 미리 계산.
    step_size = int(slide_step / 0.48)
    segment_results = {}
    scores_frame_list = []
    # 예시로 각 구간에 대해 T/F를 임의로 결정하는 로직 (이 부분은 실제 분석 로직에 따라 변경)
    for i in range(total_window_size):
        start_time = i * 0.48  # 구간의 시작 시간 (0.48초씩 밀리기 때문에)
        end_time = start_time + 0.96  # 구간의 끝 시간 (frame_length 만큼 더함)
        # 딕셔너리 형태로 구간 저장 (구간의 시간대: T/F)
        key = f"{start_time:.2f}-{end_time:.2f}"
        score = scores[i,:]  # 현재 구간에 대한 score 벡터
        if bruxism:
            is_true,score_frame= one_frame_judgment_only_bruxism(score,class_combinations,"이갈이")
        elif cough:
            is_true,score_frame=one_frame_judgment_only_cough(score,class_combinations,"기침")
        elif snoring:
            is_true,score_frame = one_frame_judgment(score,class_combinations,"코골이")
        elif speech:
            is_true,score_frame = one_frame_judgment(score,class_combinations,"잠꼬대")
        else:
            is_true,score_frame = one_frame_judgment(score,class_combinations)
        segment_results[key] = is_true
        scores_frame_list.append(score_frame)
    # 딕셔너리 형태로 저장된 구간 출력
    # if bruxism:
    #   for key, value in segment_results.items():
    #       print(f"{key}: {'T' if value else 'F'}")
    scores_group_list_temp = [sum(scores_frame_list[i:i+group_size]) for i in range(len(scores_frame_list) - group_size + 1)]
    scores_group_list = []
    # 이제 group_size 만큼 구간을 묶어서 판단하기
    for i in range(0, total_group_size*step_size,step_size):
        group = list(segment_results.items())[i:i+group_size]  # group_size 구간 묶기
        start_time = group[0][0].split("-")[0]
        end_time = group[-1][0].split("-")[-1]
        true_count = sum(1 for r in group if r[1])  # T의 개수 카운트
        # T의 개수가 min_valid_count 이상이면 해당 구간을 T로 판단
        if true_count >= min_valid_count:
            # print(f"start_time: {start_time} ~  end_time: {end_time}: T")
            scores_group_list.append(scores_group_list_temp[i])
            result_count+=1
        else:
            continue
            # print(f"start_time: {start_time} ~  end_time: {end_time}: F")
    # print(f"총 {total_group_size}개 중에 True갯수 : {result_count}, False 갯수 :{total_group_size-result_count} 최소갯수 : {required_true_count}")
    # show(global_waveform, scores, global_spectrogram_np, global_scores_np, "asdasd",segment_results)
    # 여기는 해당 이벤트인지아닌지 최종 T/F를 판단하여 보내는 부분, 즉 이미 특정이벤트라고 판단되었으면
    # 최소그룹 갯수만큼 sort시키고 더한값을 보내면 그만이다 즉 score를보낸다 !
    if (result_count >=required_true_count):
        sorted_scores = sorted(scores_group_list, reverse=True)
        return True,result_count,sum(sorted_scores[:required_true_count])
    else:
        return False,result_count,0
# global_waveform = ""
# global_spectrogram_np = ""
# global_scores_np = ""
def start(audio_path,scores, class_combinations_list, group_size_list, min_valid_count_list, required_true_count_list,waveform,spectrogram_np,scores_np):
    # global global_waveform,global_spectrogram_np,global_scores_np
    # global_waveform = waveform
    # global_spectrogram_np = spectrogram_np
    # global_scores_np = scores_np
    result_dict = {
        "이갈이":  False,
        "기침":  False,
        "코골이":  False,
        "잠꼬대":  False,
        "정상" :  False,
    }
    #순서는 이갈이,기침,코골이,잠꼬대 순서이다.
    result_scores= []
    # 각각의 항목에 대해 final_logic을 실행하여 결과 저장
    for i, category in enumerate(result_dict.keys()):
        if i == 4:
          break
        if category =="이갈이":
            judge,result_count,score= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores,
                                      bruxism=True)
            result_scores.append(score)

        elif category =="기침":
            judge,result_count,score= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores,
                                      cough =True)
            result_scores.append(result_count)
        elif category =="코골이":
            judge,result_count,score= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores,
                                      snoring =True)
            result_scores.append(score)
        elif category =="잠꼬대":
            judge,result_count,score= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores,
                                      speech =True)
            result_scores.append(score)
        else:
            judge,result_count,score= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores)
            result_scores.append(score)
        # 결과를 딕셔너리에 저장
        result_dict[category] = judge
    result_list = [int(result_dict[key]) for key in result_dict]

    # 모든 값이 False면 마지막 요소를 1로 설정
    if sum(result_list) == 0:
        result_dict['정상'] = True
        result_list.append(1)
    else:
        result_list.append(0)
    # print(result_scores)
    return result_dict,result_scores


def look_from_data(wav_data, sample_rate):
    # sample_rate, wav_data는 이미 전달된 상태

    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)  # 리샘플링
    # 만약 wav파일이 스테레오라면 스테레오 데이터를 모노로 변환
    if len(wav_data.shape) == 2:  # 스테레오 파일일 경우
        wav_data = np.mean(wav_data, axis=1)  # 두 채널을 평균 내어 모노로 변환
    duration = len(wav_data) / sample_rate
    # Audio(wav_data, rate=sample_rate)

    waveform = wav_data / tf.int16.max

    # waveform = waveform/100000
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    return scores, embeddings, spectrogram, waveform, scores_np, spectrogram_np

def get_sample_rate_from_wave(input_wav):
    with wave.open(input_wav, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()  # 샘플 레이트 가져오기
    return sample_rate

def process_large_wav(input_wav, chunk_duration=20):
    global audio_total_time
    dic_reset()
    # 입력 WAV 파일 읽기
    # sample_rate = get_sample_rate_from_wave(input_wav)
    audio_data, sample_rate = librosa.load(input_wav, sr=None)
    audio_data = np.int16(audio_data * np.iinfo(np.int16).max)
    audio_duration = len(audio_data) / sample_rate

    hours, minutes, seconds  = map(int,[audio_duration// 3600, (audio_duration % 3600) // 60, (audio_duration % 3600) % 60])
    audio_total_time = f"{hours}시간 {minutes}분 {seconds}초"
    chunk_samples = sample_rate * chunk_duration
    num_chunks = len(audio_data) // chunk_samples
    results = []

    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(audio_data))
        # print(f"start_sample : {start_sample} end_sample : {end_sample} ")
        total_seconds = int(start_sample / sample_rate)
        hours, minutes, seconds  = total_seconds// 3600, (total_seconds % 3600) // 60, (total_seconds % 3600) % 60

        ##
        total_seconds = hours * 3600 + minutes * 60 + seconds
        research_value = total_seconds//20
        block_start = (total_seconds // 20) * 20
        block_end = block_start + 19  # 20초 범위 끝

        # HH:MM:SS 형식 변환
        start_h, start_m, start_s = block_start // 3600, (block_start % 3600) // 60, block_start % 60
        end_h, end_m, end_s = block_end // 3600, (block_end % 3600) // 60, block_end % 60
        audio_chunk = audio_data[start_sample:end_sample]

        # print(f"한 청크의 소리 길이 : {len(audio_chunk) / sample_rate}")

        scores, embeddings, spectrogram, waveform, scores_np, spectrogram_np = look_from_data(audio_chunk, sample_rate)
        # print(f"현재 {int(start_sample/sample_rate)}초 부터 {int(end_sample/sample_rate)}초하고있습니다.")
        result,result_scores = start(input_wav,scores,global_class_combinations_list,group_size_list,min_valid_count_list,required_true_count_list,waveform,spectrogram_np,scores_np)
        personal_output[int(start_sample/sample_rate)//20] = result
        for key, value in result.items():
            if value:
                # 미리 만들어뒀던 각 이벤트 리스트에 시간 및 점수 기록.
                if key != "정상":
                  # print(f"일어난 이벤트 이름 : {key},\n일어난 시간 : {start_h:02}:{start_m:02}:{start_s:02} 부터 {end_h:02}:{end_m:02}:{end_s:02}\n 그때 score값: {result_scores[numbering[key]]}")
                  # print(f"result : {result}")
                  # print(f"result_scores : {result_scores}")
                  koren_to_english[key].append([f"{start_h:02}:{start_m:02}:{start_s:02} 부터 {end_h:02}:{end_m:02}:{end_s:02}",result_scores[numbering[key]]])
                ### 임
                event_output[key].append(f"{start_h:02}:{start_m:02}:{start_s:02} 부터 {end_h:02}:{end_m:02}:{end_s:02}")
        total_output[f"{start_h:02}:{start_m:02}:{start_s:02} 부터 {end_h:02}:{end_m:02}:{end_s:02}"] = result
    sort_dic()

    return audio_data,sample_rate
def extract_audio_segment(audio_data, sample_rate,start_value):
    """
    audio_data에서 특정 시간(start_time)부터 duration만큼 오디오를 추출하는 함수
    """
    chunk_samples = sample_rate*20
    start_sample = int(chunk_samples * start_value)
    end_sample = min(int((start_value + 1) * chunk_samples), len(audio_data))

    extracted_audio = audio_data[start_sample:end_sample]
    return extracted_audio

def find_chunk_by_time(str_time,audio_data,sample_rate):
  hours, minutes, seconds = map(int, str_time.split(":"))
  total_seconds = hours * 3600 + minutes * 60 + seconds
  research_value = total_seconds//20
  block_start = (total_seconds // 20) * 20
  block_end = block_start + 19  # 20초 범위 끝

  # HH:MM:SS 형식 변환
  start_h, start_m, start_s = block_start // 3600, (block_start % 3600) // 60, block_start % 60
  end_h, end_m, end_s = block_end // 3600, (block_end % 3600) // 60, block_end % 60
  print(f"{start_h:02}:{start_m:02}:{start_s:02} 부터 {end_h:02}:{end_m:02}:{end_s:02}")
  print(personal_output[research_value])
  extracted_audio = extract_audio_segment(audio_data,sample_rate,research_value)
  return extracted_audio
def total_result_show():
  for key, value in total_output.items():
    print(f"{key}: {value}")
def event_result_show(target):
  if len(event_output[target]) !=0:
    for panel in event_output[target]:
      print(panel)
  else:
    print("존재하지 않습니다.")
def summary_result_show():
  print(f"{audio_total_time} 동안의 분석 결과")
  for key, value in event_output.items():
    print(f"{key}: {len(value)}회")

def get_high_score_event(event_name,count):
  koren_to_english ={
        "이갈이" : bruxism,
        "기침" : cough,
        "잠꼬대" : speech,
        "코골이"  : snoring
    }
  if len(koren_to_english[event_name]) < count:
    print(f"모델이 감지한 {event_name} 오디오 파일의 총 갯수가 {count}보다 작습니다. 오디오의 갯수보다 낮춰서 다시 실행해주세요.\n{event_name} 오디오 총 갯수 : {len(koren_to_english[event_name])}")
    return
  for i in range(count):
    # print(koren_to_english[event_name])
    # {koren_to_english[event_name][i][1]}
    # print(f"event_name : {koren_to_english[event_name]}")
    print(f"{i+1}번째 : {koren_to_english[event_name][i][0]}, score :  {koren_to_english[event_name][i][1]}")


def dic_reset():
    global personal_output ,total_output,bruxism,cough,speech,snoring,koren_to_english,event_output
    personal_output = {}
    total_output = {}
    bruxism = []
    cough = []
    speech = []
    snoring = []
    koren_to_english ={
        "이갈이" : bruxism,
        "기침" : cough,
        "잠꼬대" : speech,
        "코골이"  : snoring
    }
    event_output =  {"이갈이": [], "기침": [], "코골이": [], "잠꼬대": [], "정상": []}
def sort_dic():
    global bruxism,cough,speech,snoring
    bruxism = sorted(bruxism, key=lambda x: x[1], reverse=False)
    cough = sorted(cough, key=lambda x: x[1], reverse=False)
    speech = sorted(speech, key=lambda x: x[1], reverse=False)
    snoring = sorted(snoring, key=lambda x: x[1], reverse=False)