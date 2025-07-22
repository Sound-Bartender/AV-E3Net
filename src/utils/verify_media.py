import os
import torchaudio
import wave


def verify_wav_file(file_path):
    """
    주어진 경로의 WAV 파일이 유효한지 검사합니다.
    1. torchaudio로 로드 시도
    2. wave 모듈로 열기 시도
    둘 중 하나라도 실패하면 손상된 파일로 간주합니다.

    Args:
        file_path (str): 검사할 WAV 파일의 경로

    Returns:
        bool: 파일이 유효하면 True, 그렇지 않으면 False
    """
    # 1. torchaudio를 사용한 검증 (오류가 발생했던 방식)
    try:
        # torchaudio.load는 내부적으로 FFmpeg를 사용하여 파일을 디코딩합니다.
        # 여기서 오류가 발생하면 파일 데이터에 문제가 있을 확률이 높습니다.
        _ = torchaudio.load(file_path)
    except Exception as e:
        print(f"❌ [torchaudio Error] 파일 로드 실패: {file_path}")
        print(f"   오류 메시지: {e}")
        return False

    # 2. Python 기본 wave 모듈을 사용한 검증
    try:
        # wave 모듈은 WAV 파일의 헤더 정보를 읽고 기본적인 구조를 확인합니다.
        # 헤더가 손상된 경우 여기서 오류가 발생할 수 있습니다.
        with wave.open(str(file_path), 'rb') as wf:
            _ = wf.getnframes()
    except Exception as e:
        print(f"❌ [wave Error] 파일 열기 실패: {file_path}")
        print(f"   오류 메시지: {e}")
        return False

    return


def verify_video_file(self, video_path: str) -> bool:
    """
    주어진 경로의 비디오 파일이 유효하고 프레임을 포함하고 있는지 확인합니다.

    Args:
        video_path (str): 검사할 비디오 파일의 경로

    Returns:
        bool: 비디오가 유효하면 True, 그렇지 않으면 False를 반환합니다.
    """
    try:
        # 1. 파일이 실제로 존재하는지 확인
        if not os.path.exists(video_path):
            # print(f"오류: '{video_path}' 파일이 존재하지 않습니다.")
            return False

        # 2. 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)

        # 3. 비디오가 성공적으로 열렸는지 확인
        if not cap.isOpened():
            # print(f"오류: '{video_path}' 비디오를 열 수 없습니다. (손상되었거나 지원하지 않는 형식)")
            cap.release()
            return False

        # 4. 비디오에서 첫 프레임을 읽을 수 있는지 확인
        ret, frame = cap.read()
        if not ret:
            # print(f"오류: '{video_path}'에서 프레임을 읽을 수 없습니다. (비어있는 파일)")
            cap.release()
            return False

        # 모든 검사를 통과하면 비디오는 유효함
        cap.release()
        return True

    except Exception as e:
        # 예상치 못한 다른 오류 (ValueError 포함) 발생 시
        # print(f"오류 발생: {e}")
        if 'cap' in locals():
            cap.release()
        return False