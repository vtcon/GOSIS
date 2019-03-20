#pragma once

#include <QDialog>
#include "ui_DocumentationDialog.h"

class DocumentationDialog : public QDialog, public Ui::DocumentationDialog
{
	Q_OBJECT

public:
	DocumentationDialog(QWidget *parent = Q_NULLPTR);
	~DocumentationDialog();
};
