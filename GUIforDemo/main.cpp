#include "GUIforDemo.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	GUIforDemo w;
	w.show();
	return a.exec();
}
