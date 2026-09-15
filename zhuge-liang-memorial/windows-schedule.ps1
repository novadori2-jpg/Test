# 와룡상소(臥龍上疏) - Windows 작업 스케줄러 등록 스크립트
# 매일 오전 7시, 그리고 로그온 시 상소문 웹페이지를 기본 브라우저로 엽니다.
#
# 사용법: PowerShell을 관리자 권한으로 열고
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\windows-schedule.ps1

$Url = "https://claude.ai/artifact/NWYuVYrc7PpLkodXc4ugYP"
$TaskNameDaily = "와룡상소-매일7시"
$TaskNameLogon = "와룡상소-로그온시"

# 기본 브라우저로 URL을 여는 명령 (rundll32 방식 — cmd/start의 따옴표 이슈가 없어 더 안정적)
Get-ScheduledTask -TaskName $TaskNameDaily -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
Get-ScheduledTask -TaskName $TaskNameLogon -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
$Action = New-ScheduledTaskAction -Execute "rundll32.exe" -Argument "url.dll,FileProtocolHandler $Url"

# 1) 매일 오전 7시
$TriggerDaily = New-ScheduledTaskTrigger -Daily -At 7:00AM
Register-ScheduledTask -TaskName $TaskNameDaily -Action $Action -Trigger $TriggerDaily -Description "매일 오전 7시, 와룡상소를 띄웁니다." -Force

# 2) 로그온(컴퓨터를 켤 때/로그인할 때) 시
$TriggerLogon = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName $TaskNameLogon -Action $Action -Trigger $TriggerLogon -Description "로그온 시, 와룡상소를 띄웁니다." -Force

Write-Host "등록 완료: '$TaskNameDaily', '$TaskNameLogon' 두 작업이 작업 스케줄러에 등록되었습니다."
Write-Host "확인/삭제: 작업 스케줄러(taskschd.msc)에서 위 이름으로 찾을 수 있습니다."
Write-Host "삭제하려면: Unregister-ScheduledTask -TaskName '$TaskNameDaily' -Confirm:`$false"
Write-Host "           Unregister-ScheduledTask -TaskName '$TaskNameLogon' -Confirm:`$false"
