#!/bin/bash
# 와룡상소(臥龍上疏) - Linux crontab 등록 스크립트
# 매일 오전 7시, 그리고 로그인/재부팅 시 상소문 웹페이지를 기본 브라우저로 엽니다.

set -e

URL="https://claude.ai/code/artifact/ae3024b3-c8f7-414d-84e2-5142463ff948"

# 어떤 명령으로 브라우저를 열지 결정 (xdg-open 을 우선 사용)
if command -v xdg-open >/dev/null 2>&1; then
  OPEN_CMD="xdg-open"
elif command -v gio >/dev/null 2>&1; then
  OPEN_CMD="gio open"
else
  echo "오류: xdg-open 또는 gio 명령을 찾을 수 없습니다. 데스크톱 환경이 필요합니다." >&2
  exit 1
fi

MARKER="# waryong-sangso"
CRON_DAILY="0 7 * * * DISPLAY=:0 $OPEN_CMD \"$URL\" $MARKER-daily"
CRON_REBOOT="@reboot sleep 30 && DISPLAY=:0 $OPEN_CMD \"$URL\" $MARKER-reboot"

# 기존 등록 제거 후 새로 추가 (중복 방지)
( crontab -l 2>/dev/null | grep -v "$MARKER" ; echo "$CRON_DAILY" ; echo "$CRON_REBOOT" ) | crontab -

echo "등록 완료: crontab 에 아래 두 줄이 추가되었습니다."
echo "  $CRON_DAILY"
echo "  $CRON_REBOOT"
echo ""
echo "확인: crontab -l"
echo "삭제하려면: crontab -l | grep -v '$MARKER' | crontab -"
echo ""
echo "※ DISPLAY=:0 은 그래픽 로그인 세션 번호를 가정한 값입니다. 다르면 'echo \$DISPLAY' 로 확인해 바꿔주세요."
