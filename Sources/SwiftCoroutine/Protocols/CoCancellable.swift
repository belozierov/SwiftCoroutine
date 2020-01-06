//
//  CoCancellable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 04.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Combine

public protocol CoCancellable: Cancellable {
    
    func cancel()
    
}
