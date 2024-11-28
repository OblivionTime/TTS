import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request
import soundfile as sf
app = Flask(__name__)
from GPT_SoVITS.TTS import TTS

gpt_path = os.environ.get(
    "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")

tts=TTS()
tts.change_gpt_weights(gpt_path=gpt_path)
tts.change_sovits_weights(sovits_path=sovits_path)
def synthesize(mode, target_text, target_path, special=False):
    # Change model weights
    ref_audio_path = os.path.join("model", mode + ".wav")
    ref_audio_text_path=os.path.join("model", mode + ".txt")
    prompt_text=""
    with open(ref_audio_text_path,"r",encoding="utf-8") as fs:
        prompt_text=fs.read()
    # Synthesize audio
    synthesis_result = tts.get_tts_wav(ref_wav_path=ref_audio_path,
                                   prompt_text=prompt_text,
                                   text=target_text,
                                   top_p=1, temperature=1,
                                   )

    result_list = list(synthesis_result)
    if result_list:
        last_sampling_rate,last_audio_data = result_list[-1]
        output_wav_path = target_path
        if not special:
            # 获取当前的UTC时间作为datetime对象
            current_datetime_utc = datetime.now(timezone.utc)
            # 将datetime对象转换为时间戳
            current_timestamp = current_datetime_utc.timestamp()
            current_timestamp_seconds = int(current_timestamp)
            filename = f'{current_timestamp_seconds}.wav'
            output_wav_path = os.path.join(target_path, filename)
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
        if not special:
            return filename
        return ""


def check_file_exists(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False


@app.route('/api/generateSpeech', methods=['POST'])
def generateSpeech():
    try:
        # 获取POST请求中的JSON数据
        data = request.get_json()

        # 提取数据中的"mode"和"message"字段
        mode = data.get('mode')
        message = data.get('message')
        target_path = data.get("target_path")
        if not check_file_exists(os.path.join("model", mode + ".wav")):
            return jsonify({
                "code": 40001,
                "message": "模型不存在,请输入存在的模型"
            })
        output_wav_path = synthesize(mode, message, target_path, special=os.path.isfile(target_path))
        return jsonify({
            "code": 0,
            "file_name": output_wav_path
        })

    except Exception as e:
        # 如果出现异常，返回错误响应给客户端
        error_response = {
            'code': 40001,
            'message': f'接收数据时出现错误: {e}'
        }
        return jsonify(error_response)



if __name__ == '__main__':
    app.run(debug=True)
