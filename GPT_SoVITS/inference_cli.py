import argparse
import os
from GPT_SoVITS.TTS import TTS
import soundfile as sf
gpt_path = os.environ.get(
    "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
tts=TTS()
tts.change_gpt_weights(gpt_path=gpt_path)
tts.change_sovits_weights(sovits_path=sovits_path)
def synthesize(mode, target_text, target_path, special=False):
    # Change model weights

    ref_audio_path = os.path.join("model", mode + ".wav")
    # Synthesize audio
    synthesis_result = tts.get_tts_wav(ref_wav_path=ref_audio_path,
                                   prompt_text="在一无所知中,梦里的一天结束了,一个新的轮回便会开始。",
                                   text=target_text,
                                   top_p=1, temperature=1,
                                   )
    result_list = list(synthesis_result)
    if result_list:
        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path =  target_path
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            print(f"Audio saved to {output_wav_path}")
    print("运行完成")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    parser.add_argument('--ref_language', required=True, choices=["中文", "英文", "日文"], help="Language of the reference audio")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--target_language', required=True, choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], help="Language of the target text")
    parser.add_argument('--output_path', required=True, help="Path to the output directory")

    args = parser.parse_args()

    synthesize(args.gpt_model, args.sovits_model, args.ref_audio, args.ref_text, args.ref_language, args.target_text, args.target_language, args.output_path)

if __name__ == '__main__':
    synthesize("派蒙","你好啊!!","1.wav")
    synthesize("派蒙","主人,下午好!今天徐家汇当前气温15.1℃,最高气温:15.0℃,最低气温:10.0℃,晴未来一周大部分时间天气晴好,未来一周最高气温和最低气温温差较大,请注意保暖。","2.wav")

