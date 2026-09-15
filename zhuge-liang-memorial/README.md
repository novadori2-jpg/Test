# 와룡상소 — 매일 아침 제갈량의 상소문 띄우기

매일 오전 7시, 또는 컴퓨터를 켤 때마다 제갈량(諸葛亮) 말투로 쓴 상소문 창이
자동으로 뜨도록 설정하는 방법을 정리했습니다.

## 상소문 자체

상소문은 아래 두루마리형 웹페이지(Artifact)로 만들어 두었습니다.

**👉 상소문 링크:** https://claude.ai/artifact/NWYuVYrc7PpLkodXc4ugYP

> ⚠️ 링크가 이전에 안내드린 `claude.ai/code/artifact/...` 주소에서 위 주소로
> **바뀌었습니다.** 매일 새 글을 저장하는 기능(DB)을 추가하면서 주소가 새로
> 발급되었기 때문입니다. Windows 작업 스케줄러 등에 이전 링크를 등록해
> 두셨다면 반드시 위 새 링크로 다시 등록해 주세요.

- 매일 오전 상소가 새로 지어져 이 페이지의 DB에 쌓이고, 페이지 상단의
  "지난 상소문" 목록에서 과거 글도 다시 읽을 수 있습니다.
- 아직 오늘 글이 도착하기 전이거나 자동 생성이 처음 설정되는 중에는, 가장
  최근 글(또는 최초 원고)을 대신 보여줍니다.
- 본문 텍스트 최초 원고는 `sangso-content.md`에도 보관해 두었습니다(백업용).
- 페이지 하단에는 매일 인장을 찍어 "몇 일째 상소를 받들고 있는지" 기록하는
  작은 장치도 있습니다(브라우저 로컬 저장소 기준이라 기기를 바꾸면 초기화됩니다).
- 이 아티팩트는 로그인한 사용자만 볼 수 있는 비공개 상태입니다 — 처음 여는
  브라우저에서 claude.ai에 한 번 로그인해 두시면 그다음부터는 로그인 화면 없이
  바로 보입니다.

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
