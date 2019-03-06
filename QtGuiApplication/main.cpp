#include "QtGuiApplication.h"
#include <QtWidgets/QApplication>

#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <windows.h>

int main(int argc, char *argv[])
{
	{
		AllocConsole();
		AttachConsole(GetCurrentProcessId());
		freopen("CON", "w", stdout);
	}

	QApplication a(argc, argv);
	QtGuiApplication w;
	w.show();
	return a.exec();
}
