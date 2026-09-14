#!/bin/bash
# 와룡상소(臥龍上疏) - macOS 설치 스크립트
# com.waryong.sangso.plist 를 ~/Library/LaunchAgents 에 등록합니다.

set -e

PLIST_NAME="com.waryong.sangso.plist"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$HOME/Library/LaunchAgents"
DEST_PATH="$DEST_DIR/$PLIST_NAME"

if [ ! -f "$SRC_DIR/$PLIST_NAME" ]; then
  echo "오류: $PLIST_NAME 파일을 찾을 수 없습니다. 같은 폴더에 두고 실행해 주세요." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_DIR/$PLIST_NAME" "$DEST_PATH"

# 기존에 로드되어 있으면 내렸다가 다시 올림
launchctl unload "$DEST_PATH" 2>/dev/null || true
launchctl load "$DEST_PATH"

echo "등록 완료: 매일 오전 7시와 로그인 시 와룡상소가 기본 브라우저로 열립니다."
echo "해제하려면 다음을 실행하세요:"
echo "  launchctl unload $DEST_PATH && rm $DEST_PATH"
