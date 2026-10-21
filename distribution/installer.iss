#define MyAppName "Subtitle-AUTO"
#define MyAppVersion "0.1.0-preview"
#define MyAppPublisher "Subtitle-AUTO contributors"
[Setup]
AppId={{F62B981A-9B82-48A6-B9AD-11F0A471EC24}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Subtitle-AUTO
DefaultGroupName=Subtitle-AUTO
OutputDir=dist
OutputBaseFilename=Subtitle-AUTO-Setup-0.1.0-preview
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\README.md
WizardStyle=modern
[Files]
Source: "..\*.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\*.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\ENCODING_POLICY.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "bootstrap.py"; DestDir: "{app}\distribution"; Flags: ignoreversion
Source: "config.example.json"; DestDir: "{app}\distribution"; Flags: ignoreversion
[Icons]
Name: "{group}\Subtitle-AUTO documentation"; Filename: "{app}\README.md"
[Run]
Filename: "notepad.exe"; Parameters: """{app}\README.md"""; Description: "설치 안내 읽기"; Flags: postinstall nowait skipifsilent
