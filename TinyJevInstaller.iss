#define MyAppName "TinyJev"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Halit Eren Ozkir"
#define MyAppExeName "TinyJev.exe"

[Setup]
AppId={{2E9348BE-31F2-489A-89EA-AECC9B0E6AFD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\TinyJev
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename=TinyJev-Setup-0.1.0
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\TinyJev\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\TinyJev"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\TinyJev"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch TinyJev"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C reg delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v TinyJevServerGUI /f"; Flags: runhidden; RunOnceId: "tinyjev-remove-run-key"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
