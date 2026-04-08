cd tools\RasMapperStoreMap
dotnet build -c Release
copy bin\Release\net472\RasMapperStoreMap.exe ..\..\src\rivia\bin\
cd ..\..