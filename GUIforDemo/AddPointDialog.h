#pragma once

#include <QDialog>
#include "ui_AddPointDialog.h"

class AddPointDialog : public QDialog, public Ui::AddPointDialog
{
	Q_OBJECT

public:
	AddPointDialog(QWidget *parent = Q_NULLPTR);
	~AddPointDialog();
};
