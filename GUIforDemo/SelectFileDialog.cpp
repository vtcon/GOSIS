#include "SelectFileDialog.h"

SelectFileDialog::SelectFileDialog(QWidget *parent)
	: QFileDialog(parent)
{
	setupUi(this);
}

SelectFileDialog::~SelectFileDialog()
{
}
