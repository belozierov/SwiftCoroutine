//
//  Coroutine+Delay.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 04.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

extension Coroutine {
    
    @inlinable public static func delay(_ sec: Double) throws {
        try delay(.now() + .milliseconds(Int(sec * 1000)))
    }
    
    @inlinable public static func delay(_ time: DispatchTime) throws {
        let coroutine = try current()
        let timer = DispatchSource.createTimer(timeout: time, handler: coroutine.resume)
        coroutine.suspend(with: timer.activate)
    }
    
}
