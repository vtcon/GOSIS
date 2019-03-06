#include "ConsoleOut.h"

ConsoleOut::ConsoleOut(QWidget *parent)
	: QWidget(parent)
{
	setupUi(this);

	m_qd = new QDebugStream(std::cout, textEdit);
}

ConsoleOut::~ConsoleOut()
{
}
