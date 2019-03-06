#pragma once

#include <QWidget>
#include "ui_ConsoleOut.h"
#include "QDebugStream.h"

class ConsoleOut : public QWidget, public Ui::ConsoleOut
{
	Q_OBJECT

public:
	ConsoleOut(QWidget *parent = Q_NULLPTR);
	~ConsoleOut();

private:
	QDebugStream* m_qd;
};
