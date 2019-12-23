//
//  Dispatcher+Scheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Combine

@available(OSX 10.15, iOS 13.0, *)
extension Dispatcher {
    
    @inlinable static func scheduler<S: Scheduler>(scheduler: S) -> Dispatcher {
        Dispatcher(dispatcher: scheduler.schedule)
    }
    
    @inlinable static func scheduler<S: Scheduler>(scheduler: S, options: S.SchedulerOptions?) -> Dispatcher {
        Dispatcher { scheduler.schedule(options: options, $0) }
    }
    
}
