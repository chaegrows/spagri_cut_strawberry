import os
import shutil

def copy_contents(src_dir, dest_dir):
    """
    src_dir 내부의 모든 파일/폴더를 dest_dir로 복사합니다.
    src_dir라는 상위 폴더를 만들지 않고, 그 안의 내용물만 복사합니다.
    """
    items = os.listdir(src_dir)
    for item in items:
        s = os.path.join(src_dir, item)  # 원본
        d = os.path.join(dest_dir, item) # 대상
        if os.path.isdir(s):
            # 폴더인 경우 copytree
            if not os.path.exists(d):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                print(f"주의: 이미 존재하는 폴더이므로 덮어쓰기가 필요할 수 있습니다: {d}")
        else:
            shutil.copy2(s, d)

def main(source_paths):
    # BASE_DIR = "/opt/config/common/specifications"  # 상황에 맞춰 수정
    # SERIAL_DIR = "/opt/config/common"  # 상황에 맞춰 수정
    BASE_DIR = "/home/metafarmers/mf_project_test/common/config/common/specifications"  # 상황에 맞춰 수정
    SERIAL_DIR = "/home/metafarmers/mf_project_test/common/config/common"  # 상황에 맞춰 수정

    # 복사 목적지: spec_type에 따른 최종 경로
    destination_paths = {
        "farm": os.path.join(BASE_DIR, "farm", "default"),       # 디렉터리 (내용물만 복사)
        "job": os.path.join(BASE_DIR, "job", "default.yaml"),   # 단일 파일
        "robot": os.path.join(BASE_DIR, "robot", "assemble.yaml"),
        "serial_info": os.path.join(SERIAL_DIR, "serial_info.yaml") # 단일 파일 예시
    }
    for dest in destination_paths.values():
        print(dest)
        # os.chmod(dest, 0o666)
    # 복사 원본: 각 spec_type에 대해 어디서 복사할 것인지 사전에 정의
    #  - farm은 폴더 이름 (BASE_DIR/farm/XXX)
    #  - job, robot, serial_info는 파일 이름 (BASE_DIR/<spec_type>/XXX.yaml)
    # 필요에 맞게 수정하세요.

    spec_types = ["farm", "job", "robot", "serial_info"]

    for spec_type in spec_types:
        # 목적지
        dest = destination_paths[spec_type]

        # 원본( farm은 폴더, 나머지는 파일 )
        src_name = source_paths[spec_type]

        if spec_type == "farm":
            # farm인 경우: BASE_DIR/farm/src_name 폴더의 **내용물**을 farm/default 폴더에 복사
            full_src_path = os.path.join(BASE_DIR, "farm", src_name)
            if not os.path.isdir(full_src_path):
                print(f"[farm] 에러: 폴더가 존재하지 않습니다: {full_src_path}")
                continue

            os.makedirs(dest, exist_ok=True)
            copy_contents(full_src_path, dest)
            print(f"[farm] {full_src_path} 내부 내용 -> {dest} 복사 완료.")

        else:
            # job, robot, serial_info는 단일 파일 복사
            # 파일 경로: BASE_DIR/<spec_type>/src_name
            if spec_type == "serial_info":
                # 만약 serial_info의 파일이 실제로 BASE_DIR/serial_info/ 에 있는 게 아니라면
                # 원하는 폴더 구조에 맞춰 아래를 수정
                full_yaml_path = os.path.join(SERIAL_DIR, src_name)
            else:
                full_yaml_path = os.path.join(BASE_DIR, spec_type, src_name)

            if not os.path.isfile(full_yaml_path):
                print(f"[{spec_type}] 에러: 파일이 존재하지 않습니다: {full_yaml_path}")
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(full_yaml_path, dest)
            print(f"[{spec_type}] {full_yaml_path} -> {dest} 덮어쓰기 완료.")

if __name__ == "__main__":
    source_paths = {
    "farm": "metafarmers250401",
    "job": "isu_250409.yaml",
    "robot": "assemble_rb.yaml",
    "serial_info": "serial_info_isu.yaml"
    }

    main(source_paths)
