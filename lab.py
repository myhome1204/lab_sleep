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
    "잠꼬대" :2,
    "코골아" :3

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
    {500: (0.1, 0.8), 439: (0.1, 0.7), 50: (0.1, 0.6)},  # Inside small room, Splash, Biting

    # 박수치는 소리 포함 이갈이
    {399: (0.1, 0.9), 403: (0.1, 0.8), 410: (0.1, 0.7)},  # Ratchet, pawl, Gears, Camera

    # 과자 씹는 소리 포함 이갈이
    {410: (0.1, 0.9), 398: (0.1, 0.8), 435: (0.1, 0.7)}   # Camera, Mechanisms, Glass
    ],

    #기침 조합입니다
    [
    {42: (0.1, 1.0)},  # 기침 단독
    {42: (0.1, 0.7), 43: (0.2, 0.5)},  # 기침 + 목 가다듬기(Throat clearing)
    {42: (0.1, 0.6), 44: (0.2, 0.5)},  # 기침 + 재채기(Sneeze)
    {42: (0.1, 0.6), 36: (0.2, 0.5)},  # 기침 + 숨소리(Breathing)
    {42: (0.1, 0.8), 23: (0.2, 0.5)},  # 기침 + 한숨(Sigh)
    {42: (0.1, 0.7), 33: (0.2, 0.5)},  # 기침 + 신음(Groan)
    {42: (0.1, 0.6), 39: (0.2, 0.5)},  # 기침 + 헉(Gasp)
    {42: (0.1, 0.7), 55: (0.2, 0.4)},  # 기침 + 방귀(Fart)
    {42: (0.1, 0.7), 49: (0.2, 0.5)},  # 기침 + 씹는 소리(Chewing)
    {42: (0.1, 0.6), 41: (0.2, 0.5)},   # 기침 + 코골이(Snort)
    {42 : (0.1,0.6), 0 :(0.1,0.5)} # 기침 +speech
    ],
    #코골이 조합입니다.
    [
    {38: (0.8, 1.0)},
    {38: (0.8, 0.7), 23: (0.2, 0.8)},
    # {23: (0.2, 0.7), 34: (0.2, 0.5), 36: (0.1, 0.4)},
    # {23: (0.4, 0.6), 33: (0.2, 0.5), 34: (0.2, 0.5)},
    # {23: (0.4, 0.7), 36: (0.2, 0.5), 55: (0.1, 0.4)}
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
    4,
    #코골이 그룹 사이즈 입니다
    4,
    #잠꼬대 그룹 사이즈 입니다.
    4,
]
min_valid_count_list = [
    #이갈이 그룹 T/F 판단 최소값 입니다
    2,
    #기침 그룹 T/F 판단 사이즈 입니다
    2,
    #코골이 그룹 T/F 판단 사이즈 입니다
    2,
    #잠꼬대 그룹 T/F 판단 이즈 입니다.
    2,
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
def is_cough_in_top_5(scores):
    """ 상위 5개 클래스 중 Cough(42)가 포함되어 있는지 확인 """
    top_5_indices = np.argsort(scores)[::-1][:10]  # 확률값 기준 상위 5개 클래스 추출
    # get_class_name(class_id)
    # for i in top_5_indices:

    #     print(f"기침 상위권 목록 {get_class_name(i)}")
    return 42 in top_5_indices  # Cough(42)가 상위 5개 안에 있는지 확인

def one_frame_judgment_only_cough(scores, class_combinations_list):
    """ 기존 조합 기반 판단 + Cough(42) 포함 여부 추가 """
    temp1 = one_frame_judgment(scores, class_combinations_list)
    # top_5_indices = np.argsort(scores)[::-1][:20]  # 확률값 기준 상위 5개 클래스 추출
    # get_class_name(class_id)
    # for i in top_5_indices:
    #     print(f"기침 상위권 목록 {get_class_name(i)}")
    # print("\n\n")
    # 기존 조합 검사
    if temp1:
        return True  # 기존 조합이 충족되면 바로 True

    temp2 = is_cough_in_top_5(scores)  # Cough(42) 상위 5개 내 포함 여부 검사
    if temp2:
        return True  # 기침이 상위 5개 안에 있으면 True

    return False  # 둘 다 아니라면 False
# 한 프레임에 대한 T/F 계산 함수 , scores는 yamnet의 한 프레임에 대한 확률벡터값(521,1)
def one_frame_judgment(scores, class_combinations):
    for combination in class_combinations:
        all_match = True  # 현재 combination이 완전히 만족하는지 확인하는 변수

        for class_id, score_range in combination.items():
            if len(score_range) >=3:
                print(score_range)
            min_score, max_score = score_range  # 범위로부터 최소값, 최대값 분리
            score = scores[class_id]
            # print(f"{get_class_name(class_id)}의 score값은 현재 : {score}이다")
            if not (min_score <= score <= max_score):  # 하나라도 조건을 만족 못 하면 실패
                all_match = False
                break
        if all_match:  # 해당 combination이 완벽하게 만족하면 True 반환
            return True
    return False  # 모든 조합이 실패하면 False 반환


# 오디오 파일에 대한 최종 T/F 판단함수 , audio_path는 오디오 파일, class_combinations 은 조건 , group_size는 내가 묶을 단위의 frame숫자,
# min_valid_count 는 group에서 몇개이상일지 판단하는 값, required_true_count는 실제 최종판단해서 true 갯수 기준 , scores 는 Yamnet의 scores.numpy
def final_logic(class_combinations, group_size, min_valid_count, required_true_count,scores,slide_step=0.48,cough=False):
    real_time = group_size_dic[group_size]  # 오디오 파일 20초로 고정
    result_count = 0
    total_window_size = 40 # 판단해야하는 총 frame수 40으로 고정
    # 구간별 T/F 결과를 딕셔너리 형태로 저장할 리스트 (각 구간에 대한 T/F 값)
    segment_results = {}

    # 예시로 각 구간에 대해 T/F를 임의로 결정하는 로직 (이 부분은 실제 분석 로직에 따라 변경)
    for i in range(total_window_size):
        start_time = i * 0.48  # 구간의 시작 시간 (0.48초씩 밀리기 때문에)
        end_time = start_time + 0.96  # 구간의 끝 시간 (frame_length 만큼 더함)
        # 딕셔너리 형태로 구간 저장 (구간의 시간대: T/F)
        key = f"{start_time:.2f}-{end_time:.2f}"

        score = scores[i,:]  # 현재 구간에 대한 score 벡터
        if cough:
            is_true=one_frame_judgment_only_cough(score,class_combinations)
        else:
            is_true = one_frame_judgment(score,class_combinations)
        segment_results[key] = is_true
    # 딕셔너리 형태로 저장된 구간 출력
    # for key, value in segment_results.items():
    #     print(f"{key}: {'T' if value else 'F'}")
    total_group_size  = total_group_size_calculate_dic[group_size] #그룹사이즈 미리 계산.
    step_size = int(slide_step / 0.48)

    # 이제 group_size 만큼 구간을 묶어서 판단하기
    for i in range(0, total_group_size*step_size,step_size):
        group = list(segment_results.items())[i:i+group_size]  # group_size 구간 묶기
        start_time = group[0][0].split("-")[0]
        end_time = group[-1][0].split("-")[-1]
        true_count = sum(1 for r in group if r[1])  # T의 개수 카운트
        # T의 개수가 min_valid_count 이상이면 해당 구간을 T로 판단
        if true_count >= min_valid_count:
            # print(f"start_time: {start_time} ~  end_time: {end_time}: T")
            result_count+=1
        else:
            continue
            # print(f"start_time: {start_time} ~  end_time: {end_time}: F")
    # print(f"총 {total_group_size}개 중에 True갯수 : {result_count}, False 갯수 :{total_group_size-result_count} 최소갯수 : {required_true_count}")
    # show(global_waveform, scores, global_spectrogram_np, global_scores_np, "asdasd",segment_results)
    if (result_count >=required_true_count):
        return True,result_count
    else:
        return False,result_count
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
        if category =="기침":
          judge,result_count= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores,
                                      cough =True)
          result_scores.append(result_count)
        else:
          judge,result_count= final_logic(class_combinations=class_combinations_list[i],
                                      group_size=group_size_list[i],
                                      min_valid_count=min_valid_count_list[i],
                                      required_true_count=required_true_count_list[i],
                                      scores=scores)
          result_scores.append(result_count)
        # 결과를 딕셔너리에 저장
        result_dict[category] = judge
    result_list = [int(result_dict[key]) for key in result_dict]

    # 모든 값이 False면 마지막 요소를 1로 설정
    if sum(result_list) == 0:
        result_dict['정상'] = True
        result_list.append(1)
    else:
        result_list.append(0)
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

        ##


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
  if len(koren_to_english[event_name]) < count:
    print(f"모델이 감지한 {event_name} 오디오 파일의 총 갯수가 {count}보다 작습니다. 오디오의 갯수보다 낮춰서 다시 실행해주세요.\n{event_name} 오디오 총 갯수 : {len(koren_to_english[event_name])}")
    return
  for i in range(count):
    # {koren_to_english[event_name][i][1]}
    print(f"{koren_to_english[event_name][i][0]}")
    
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
event_output = {"이갈이": [], "기침": [], "코골이": [], "잠꼬대": [], "정상": []}
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
    bruxism = sorted(bruxism, key=lambda x: x[1], reverse=True)
    cough = sorted(cough, key=lambda x: x[1], reverse=True)
    speech = sorted(speech, key=lambda x: x[1], reverse=True)
    snoring = sorted(snoring, key=lambda x: x[1], reverse=True)