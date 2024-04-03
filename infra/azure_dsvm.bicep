// Provision a Data Science Virtual Machine with JupyterHub and RStudio Server  on Azure Cloud
// NOTE: The necessary NVIDIA drivers, CUDA and popular Deep Learning frameworks are installed by default.
// Ref: https://learn.microsoft.com/en-gb/samples/azure/azure-quickstart-templates/vm-ubuntu-dsvm-gpu-or-cpu/
@description('Username for Administrator Account')
param adminUsername string

@description('The name of you Virtual Machine.')
param vmName string

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Choose between CPU or GPU processing')
@allowed([
  'CPU-4GB'
  'CPU-7GB'
  'CPU-8GB'
  'GPU-16GB'
])
param cpu_gpu string = 'GPU-16GB'

@description('Name of the VNET')
param virtualNetworkName string = 'vNet-${uniqueString(resourceGroup().id)}'

@description('Name of the subnet in the virtual network')
param subnetName string = 'subnet'

@description('Name of the Network Security Group')
param networkSecurityGroupName string = 'SecGroupNet'

@description('Type of authentication to use on the Virtual Machine. SSH key is recommended.')
@allowed([
  'sshPublicKey'
  'password'
])
param authenticationType string = 'sshPublicKey'

@description('SSH Key or password for the Virtual Machine. SSH key is recommended.')
@secure()
param adminPasswordOrKey string

var networkInterfaceName = '${vmName}NetInt'
var virtualMachineName = vmName
var publicIpAddressName = '${vmName}PublicIP'
var subnetRef = resourceId('Microsoft.Network/virtualNetworks/subnets', virtualNetworkName, subnetName)
var nsgId = networkSecurityGroup.id
var osDiskType = 'StandardSSD_LRS'
var storageAccountName = 'storage${uniqueString(resourceGroup().id)}'
var storageAccountType = 'Standard_LRS'
var storageAccountKind = 'Storage'
var vmSize = {
  'CPU-4GB': 'Standard_B2s'
  'CPU-7GB': 'Standard_D2s_v3'
  'CPU-8GB': 'Standard_D2s_v3'
  'GPU-16GB': 'Standard_NC16as_T4_v3'
}
var linuxConfiguration = {
  disablePasswordAuthentication: true
  ssh: {
    publicKeys: [
      {
        path: '/home/${adminUsername}/.ssh/authorized_keys'
        keyData: adminPasswordOrKey
      }
    ]
  }
}

resource networkInterface 'Microsoft.Network/networkInterfaces@2021-05-01' = {
  name: networkInterfaceName
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnetRef
          }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: {
            id: publicIpAddress.id
          }
        }
      }
    ]
    networkSecurityGroup: {
      id: nsgId
    }
  }
  dependsOn: [
    virtualNetwork
  ]
}

resource networkSecurityGroup 'Microsoft.Network/networkSecurityGroups@2021-05-01' = {
  name: networkSecurityGroupName
  location: location
  properties: {
    securityRules: [
      {
        name: 'JupyterHub'
        properties: {
          priority: 1010
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8000'
        }
      }
      {
        name: 'RStudioServer'
        properties: {
          priority: 1020
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8787'
        }
      }
      {
        name: 'SSH'
        properties: {
          priority: 1030
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
    ]
  }
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2021-05-01' = {
  name: virtualNetworkName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/24'
      ]
    }
    subnets: [
      {
        name: subnetName
        properties: {
          addressPrefix: '10.0.0.0/24'
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
    ]
  }
}

resource publicIpAddress 'Microsoft.Network/publicIPAddresses@2021-05-01' = {
  name: publicIpAddressName
  location: location
  sku: {
    name: 'Basic'
    tier: 'Regional'
  }
  properties: {
    publicIPAllocationMethod: 'Dynamic'
  }
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-08-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: storageAccountType
  }
  kind: storageAccountKind
}

resource virtualMachine 'Microsoft.Compute/virtualMachines@2021-11-01' = {
  name: '${virtualMachineName}-${cpu_gpu}'
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize[cpu_gpu]
    }
    storageProfile: {
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: osDiskType
        }
      }
      imageReference: {
        publisher: 'microsoft-dsvm'
        offer: 'ubuntu-2004'
        sku: '2004-gen2'
        version: 'latest'
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: networkInterface.id
        }
      ]
    }
    osProfile: {
      computerName: virtualMachineName
      adminUsername: adminUsername
      adminPassword: adminPasswordOrKey
      linuxConfiguration: ((authenticationType == 'password') ? null : linuxConfiguration)
    }
  }
}

output adminUsername string = adminUsername
