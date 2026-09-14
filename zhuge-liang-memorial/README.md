# 와룡상소 — 매일 아침 제갈량의 상소문 띄우기

매일 오전 7시, 또는 컴퓨터를 켤 때마다 제갈량(諸葛亮) 말투로 쓴 상소문 창이
자동으로 뜨도록 설정하는 방법을 정리했습니다.

## 상소문 자체

상소문 전체(클로드 코드 활용법 2,000자 이상 + 사업·삶의 방향 3,000자 이상,
총 5,000자 이상)는 아래 두루마리형 웹페이지(Artifact)로 만들어 두었습니다.

**👉 상소문 링크:** https://claude.ai/code/artifact/ae3024b3-c8f7-414d-84e2-5142463ff948

- 본문 텍스트 원본은 `sangso-content.md`에도 그대로 보관해 두었습니다(백업/수정용).
- 페이지에는 매일 인장을 찍어 "몇 일째 상소를 받들고 있는지" 기록하는 작은 장치도
  넣어 두었습니다(브라우저 로컬 저장소 기준이라 기기를 바꾸면 초기화됩니다).

## 왜 스크립트가 필요한가

지금 이 작업은 클라우드(원격) 세션에서 이루어지고 있어, 사용자의 실제 PC 전원이나
부팅 과정에 직접 손을 댈 수는 없습니다. 대신 **사용자의 컴퓨터에서 딱 한 번**
아래 스크립트를 실행해 두면, 운영체제의 "작업 스케줄러"가 매일 아침 7시 또는
로그인(부팅) 시점에 브라우저로 위 링크를 자동으로 열어 줍니다.

운영체제에 맞는 폴더/스크립트를 골라 사용하세요.

---

### Windows

1. `windows-schedule.ps1` 파일을 내려받습니다.
2. 파일 안의 `$Url` 값이 위 상소문 링크와 같은지 확인합니다.
3. PowerShell을 **관리자 권한**으로 열고 아래처럼 실행합니다.

   ```powershell
   cd 다운로드받은_폴더
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\windows-schedule.ps1
   ```

4. 작업 스케줄러(Task Scheduler)에 `와룡상소` 라는 이름으로 작업 두 개가 등록됩니다.
   - 매일 오전 7:00
   - 로그온(로그인) 시

   `제어판 → 관리 도구 → 작업 스케줄러`에서 언제든 확인·삭제할 수 있습니다.

### macOS

1. `com.waryong.sangso.plist`, `macos-setup.sh` 두 파일을 같은 폴더에 내려받습니다.
2. 터미널에서 실행합니다.

   ```bash
   cd 다운로드받은_폴더
   chmod +x macos-setup.sh
   ./macos-setup.sh
   ```

3. `launchd`(로그인 항목)에 등록되어, 매일 오전 7시와 로그인 시 기본 브라우저로
   상소문 링크가 열립니다.
4. 해제하려면:

   ```bash
   launchctl unload ~/Library/LaunchAgents/com.waryong.sangso.plist
   rm ~/Library/LaunchAgents/com.waryong.sangso.plist
   ```

### Linux

1. `linux-cron-setup.sh` 파일을 내려받습니다.
2. 실행합니다.

   ```bash
   chmod +x linux-cron-setup.sh
   ./linux-cron-setup.sh
   ```

3. crontab에 "매일 오전 7시"와 "로그인(재부팅) 시" 두 항목이 추가되어,
   기본 브라우저로 상소문 링크를 엽니다.

---

## 참고

- 세 스크립트 모두 "브라우저를 열어 상소문 웹페이지를 띄우는" 방식입니다.
  Claude Code나 클라우드 세션이 사용자 PC에 상주하며 감시하는 것이 아니라,
  운영체제 자체의 예약 기능(Task Scheduler / launchd / cron)을 한 번 등록해
  두는 것이므로, 이후에는 인터넷 연결과 브라우저만 있으면 계속 작동합니다.
- 문구를 바꾸고 싶으시면 `sangso-content.md`를 수정한 뒤, Claude Code에게
  "상소문 artifact를 이 내용으로 다시 발행해줘"라고 요청하시면 됩니다.
