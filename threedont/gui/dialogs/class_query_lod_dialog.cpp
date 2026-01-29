#include "class_query_lod_dialog.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QDialogButtonBox>

ClassQueryLodDialog::ClassQueryLodDialog(QWidget *parent)
    : QDialog(parent)
{
    lodValues << 0 << 1 << 2 << 3 << 4 << -3 << -2 << -1;

    QVBoxLayout* layout = new QVBoxLayout(this);

    label = new QLabel(QString::number(lodValues[0]), this);
    label->setAlignment(Qt::AlignCenter);
    layout->addWidget(label);

    slider = new QSlider(Qt::Horizontal, this);
    slider->setMinimum(0);
    slider->setMaximum(lodValues.size() - 1);
    slider->setTickInterval(1);
    slider->setSingleStep(1);
    slider->setValue(0);
    slider->setTickPosition(QSlider::TicksBelow);
    layout->addWidget(slider);
    connect(slider, &QSlider::valueChanged, this, &ClassQueryLodDialog::on_slider_valueChanged);

    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttonBox);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &ClassQueryLodDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &ClassQueryLodDialog::reject);

    setWindowTitle(tr("Set Class Query LOD"));
    setLayout(layout);
}

ClassQueryLodDialog::~ClassQueryLodDialog() = default;

int ClassQueryLodDialog::lodValue() const
{
    return lodValues[slider->value()];
}

void ClassQueryLodDialog::on_slider_valueChanged(int value)
{
    label->setText(QString::number(lodValues[value]));
}
